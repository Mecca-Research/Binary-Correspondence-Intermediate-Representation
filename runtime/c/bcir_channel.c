/*===- bcir_channel.c - the multi-channel lowering decision in C ------------===
 * The C twin of bcir/channels' routing seam. Host tool (parses JSON, uses libc).
 *===----------------------------------------------------------------------===*/
#include "bcir_channel.h"

#include <stdio.h>
#include <string.h>

/* --- a minimal channel.json reader (top-level object; only the routing fields) ------------ */
static const char *jws(const char *p, const char *e) {
  while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
  return p;
}
/* parse a JSON string at `p` (the opening quote) into `out`; returns just past the closing quote. */
static const char *jstr(const char *p, const char *e, char *out, size_t outn) {
  if (p >= e || *p != '"') return NULL;
  p++; size_t o = 0;
  while (p < e && *p != '"') {
    char ch = *p++;
    if ((unsigned char)ch < 0x20u) return NULL;
    if (ch == '\\' && p < e) { char esc = *p++; ch = esc == 'n' ? '\n' : esc == 't' ? '\t' : esc; }
    if (out) { if (o + 1 >= outn) return NULL; out[o++] = ch; }
  }
  if (p >= e) return NULL;
  if (out) out[o] = 0;
  return p + 1;
}
/* skip one JSON value (object/array/string/number/literal); returns just past it. */
static const char *jskip(const char *p, const char *e) {
  p = jws(p, e);
  if (p >= e) return e;
  if (*p == '"') { const char *q = jstr(p, e, NULL, 0); return q ? q : e; }
  if (*p == '{' || *p == '[') {
    char open = *p, close = (open == '{') ? '}' : ']';
    p++; int depth = 1;
    while (p < e && depth) {
      p = jws(p, e); if (p >= e) break;
      if (*p == '"') { const char *q = jstr(p, e, NULL, 0); if (!q) return e; p = q; continue; }
      if (*p == open) depth++; else if (*p == close) depth--;
      p++;
    }
    return p;
  }
  while (p < e && *p != ',' && *p != '}' && *p != ']') p++;   /* number / true / false / null */
  return p;
}
static uint32_t cap_bit(const char *s) {
  struct { const char *n; uint32_t b; } C[] = {
    {"universal", BCIR_CAP_UNIVERSAL}, {"data_parallel", BCIR_CAP_DATA_PARALLEL},
    {"reduce", BCIR_CAP_REDUCE}, {"gather", BCIR_CAP_GATHER}, {"tile", BCIR_CAP_TILE},
    {"matmul", BCIR_CAP_MATMUL}, {"stream_unit", BCIR_CAP_STREAM_UNIT},
    {"scalar_stream", BCIR_CAP_SCALAR_STREAM}, {NULL, 0}};
  for (int i = 0; C[i].n; i++) if (!strcmp(C[i].n, s)) return C[i].b;
  return 0;   /* unknown tag -> no routing match (validation is a separate concern) */
}

int bcir_channel_parse(const char *json, size_t len, bcir_channel *ch, char *diag, size_t dn) {
  if (!diag) dn = 0;
  if (!json || !ch) { if (dn) snprintf(diag, dn, "channel.json: invalid arguments"); return 1; }
  memset(ch, 0, sizeof *ch); ch->modeled = 1;
  if (dn) diag[0] = 0;
  const char *p = json, *e = json + len;
  p = jws(p, e);
  if (p >= e || *p != '{') { snprintf(diag, dn, "channel.json: expected an object"); return 1; }
  p++;
  for (;;) {
    p = jws(p, e);
    if (p < e && *p == '}') { p++; break; }
    if (p >= e) { snprintf(diag, dn, "channel.json: unterminated object"); return 1; }
    char key[48];
    p = jstr(p, e, key, sizeof key);
    if (!p) { snprintf(diag, dn, "channel.json: expected a key"); return 1; }
    p = jws(p, e);
    if (p >= e || *p != ':') { snprintf(diag, dn, "channel.json: expected ':'"); return 1; }
    p++; p = jws(p, e);
    if (!strcmp(key, "name")) p = jstr(p, e, ch->name, sizeof ch->name);
    else if (!strcmp(key, "kind")) p = jstr(p, e, ch->kind, sizeof ch->kind);
    else if (!strcmp(key, "provenance")) p = jstr(p, e, ch->provenance, sizeof ch->provenance);
    else if (!strcmp(key, "modeled")) {
      if ((size_t)(e-p)>=4u&&!strncmp(p,"true",4)){ch->modeled=1;p+=4;}
      else if ((size_t)(e-p)>=5u&&!strncmp(p,"false",5)){ch->modeled=0;p+=5;}
      else { snprintf(diag,dn,"channel.json: 'modeled' must be boolean");return 1; }
    }
    else if (!strcmp(key, "capabilities")) {
      if (p < e && *p == '[') {
        p++;
        for (;;) {
          p = jws(p, e);
          if (p < e && *p == ']') { p++; break; }
          if (p >= e) { snprintf(diag, dn, "channel.json: unterminated capabilities"); return 1; }
          char cap[32];
          p = jstr(p, e, cap, sizeof cap);
          if (!p) { snprintf(diag, dn, "channel.json: bad capability"); return 1; }
          ch->capabilities |= cap_bit(cap);
          p = jws(p, e);
          if (p < e && *p == ',') { p++; p=jws(p,e); if(p>=e||*p==']'){snprintf(diag,dn,"channel.json: trailing capability comma");return 1;} }
          else if (p >= e || *p != ']') { snprintf(diag,dn,"channel.json: expected ',' or ']'");return 1; }
        }
      } else p = jskip(p, e);
    } else p = jskip(p, e);
    if (!p) { snprintf(diag, dn, "channel.json: parse error"); return 1; }
    p = jws(p, e);
    if (p < e && *p == ',') { p++; p=jws(p,e); if(p>=e||*p=='}'){snprintf(diag,dn,"channel.json: trailing comma");return 1;} }
    else if (p >= e || *p != '}') { snprintf(diag,dn,"channel.json: expected ',' or '}'");return 1; }
  }
  if (jws(p,e) != e) { snprintf(diag,dn,"channel.json: trailing data");return 1; }
  if (!ch->name[0]) { snprintf(diag, dn, "channel.json: missing 'name'"); return 1; }
  if (!ch->kind[0]) { snprintf(diag, dn, "channel.json: missing 'kind'"); return 1; }
  return 0;
}

/* --- the routing decision (mirror bcir/channels: claim_required_caps / channel_suits / route) --- */
static int starts_with(const char *s, const char *pre) {
  size_t n = strlen(pre); return strncmp(s, pre, n) == 0;
}

uint32_t bcir_claim_required_caps(const bcir_claim *cl) {
  const char *op = cl->op;
  bcir_stride sc = cl->stride;
  uint32_t req = 0;
  if (starts_with(op, "reduce.")) req |= BCIR_CAP_REDUCE;
  if (strstr(op, "gather") || sc == BCIR_STRIDE_RANDOM) req |= BCIR_CAP_GATHER;
  if (strstr(op, "matmul")) req |= BCIR_CAP_MATMUL;
  if (sc == BCIR_STRIDE_TILE) req |= BCIR_CAP_TILE;
  if (sc == BCIR_STRIDE_UNIT) req |= BCIR_CAP_STREAM_UNIT;
  if (sc == BCIR_STRIDE_SCALAR) req |= BCIR_CAP_SCALAR_STREAM;
  if (!req) req |= BCIR_CAP_DATA_PARALLEL;            /* default: elementwise work */
  return req;
}

int bcir_channel_suits(const bcir_claim *cl, const bcir_channel *ch) {
  if (ch->capabilities) {                             /* plugin: route by declared tags */
    if (ch->capabilities & BCIR_CAP_UNIVERSAL) return 1;
    return (ch->capabilities & bcir_claim_required_caps(cl)) ? 1 : 0;
  }
  const char *op = cl->op; bcir_stride sc = cl->stride;   /* legacy per-kind rule */
  if (!strcmp(ch->kind, "cpu")) return 1;
  if (!strcmp(ch->kind, "gpu")) return 1;
  if (!strcmp(ch->kind, "memory"))
    return starts_with(op, "reduce.") || strstr(op, "gather") != NULL || sc == BCIR_STRIDE_RANDOM;
  if (!strcmp(ch->kind, "fpga"))
    return sc == BCIR_STRIDE_TILE || sc == BCIR_STRIDE_UNIT || strstr(op, "matmul") != NULL;
  if (!strcmp(ch->kind, "storage"))
    return sc == BCIR_STRIDE_UNIT || sc == BCIR_STRIDE_SCALAR;
  return 0;
}

int bcir_channel_route(const bcir_claim *cl, const bcir_channel *channels, int n) {
  uint32_t req = bcir_claim_required_caps(cl);
  int best = -1, best_rank = 0;
  for (int i = 0; i < n; i++) {
    if (!bcir_channel_suits(cl, &channels[i])) continue;
    uint32_t caps = channels[i].capabilities;
    int specialized = (caps & req) && !(caps & BCIR_CAP_UNIVERSAL);
    int rank = specialized ? 0 : 1;                  /* prefer a specialized match over a fallback */
    if (best < 0 || rank < best_rank ||
        (rank == best_rank && strcmp(channels[i].name, channels[best].name) < 0)) {
      best = i; best_rank = rank;
    }
  }
  return best;
}
