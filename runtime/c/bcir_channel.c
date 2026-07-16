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
/* Parse an ASCII JSON string at `p` into `out`; return just past its closing quote.
 * Routing identifiers are ASCII by contract.  Escapes are decoded strictly; \u escapes
 * are rejected instead of being mis-decoded as the literal character 'u'. */
static const char *jstr(const char *p, const char *e, char *out, size_t outn) {
  if (p >= e || *p != '"') return NULL;
  p++; size_t o = 0;
  while (p < e && *p != '"') {
    unsigned char uch = (unsigned char)*p++;
    char ch = (char)uch;
    if (uch < 0x20u || uch >= 0x80u) return NULL;
    if (ch == '\\') {
      if (p >= e) return NULL;
      switch (*p++) {
        case '"': ch = '"'; break; case '\\': ch = '\\'; break; case '/': ch = '/'; break;
        case 'b': ch = '\b'; break; case 'f': ch = '\f'; break; case 'n': ch = '\n'; break;
        case 'r': ch = '\r'; break; case 't': ch = '\t'; break;
        default: return NULL;
      }
    }
    if (out) { if (o + 1 >= outn) return NULL; out[o++] = ch; }
  }
  if (p >= e) return NULL;
  if (out) out[o] = 0;
  return p + 1;
}
/* Strictly skip one otherwise-unused JSON value.  The depth bound makes malicious
 * nested profile metadata a deterministic refusal instead of an unbounded stack walk. */
#define BCIR_CHANNEL_JSON_DEPTH 64
static const char *jskip_depth(const char *p, const char *e, unsigned depth);

static const char *jskip_number(const char *p, const char *e) {
  if (p < e && *p == '-') p++;
  if (p >= e) return NULL;
  if (*p == '0') p++;
  else {
    if (*p < '1' || *p > '9') return NULL;
    do { p++; } while (p < e && *p >= '0' && *p <= '9');
  }
  if (p < e && *p == '.') {
    p++;
    if (p >= e || *p < '0' || *p > '9') return NULL;
    do { p++; } while (p < e && *p >= '0' && *p <= '9');
  }
  if (p < e && (*p == 'e' || *p == 'E')) {
    p++;
    if (p < e && (*p == '+' || *p == '-')) p++;
    if (p >= e || *p < '0' || *p > '9') return NULL;
    do { p++; } while (p < e && *p >= '0' && *p <= '9');
  }
  return p;
}

static const char *jskip_array(const char *p, const char *e, unsigned depth) {
  p = jws(p + 1, e);
  if (p < e && *p == ']') return p + 1;
  for (;;) {
    p = jskip_depth(p, e, depth + 1u);
    if (!p) return NULL;
    p = jws(p, e);
    if (p < e && *p == ']') return p + 1;
    if (p >= e || *p != ',') return NULL;
    p = jws(p + 1, e);
    if (p >= e || *p == ']') return NULL;
  }
}

static const char *jskip_object(const char *p, const char *e, unsigned depth) {
  p = jws(p + 1, e);
  if (p < e && *p == '}') return p + 1;
  for (;;) {
    p = jstr(p, e, NULL, 0);
    if (!p) return NULL;
    p = jws(p, e);
    if (p >= e || *p != ':') return NULL;
    p = jskip_depth(p + 1, e, depth + 1u);
    if (!p) return NULL;
    p = jws(p, e);
    if (p < e && *p == '}') return p + 1;
    if (p >= e || *p != ',') return NULL;
    p = jws(p + 1, e);
    if (p >= e || *p == '}') return NULL;
  }
}

static const char *jskip_depth(const char *p, const char *e, unsigned depth) {
  static const char jt[] = "true", jf[] = "false", jn[] = "null";
  p = jws(p, e);
  if (p >= e || depth > BCIR_CHANNEL_JSON_DEPTH) return NULL;
  if (*p == '"') return jstr(p, e, NULL, 0);
  if (*p == '{') return jskip_object(p, e, depth);
  if (*p == '[') return jskip_array(p, e, depth);
  if ((size_t)(e - p) >= sizeof jt - 1u && !memcmp(p, jt, sizeof jt - 1u))
    return p + sizeof jt - 1u;
  if ((size_t)(e - p) >= sizeof jf - 1u && !memcmp(p, jf, sizeof jf - 1u))
    return p + sizeof jf - 1u;
  if ((size_t)(e - p) >= sizeof jn - 1u && !memcmp(p, jn, sizeof jn - 1u))
    return p + sizeof jn - 1u;
  return jskip_number(p, e);
}

static const char *jskip(const char *p, const char *e) {
  return jskip_depth(p, e, 0u);
}
static uint32_t cap_bit(const char *s) {
  struct { const char *n; uint32_t b; } C[] = {
    {"universal", BCIR_CAP_UNIVERSAL}, {"data_parallel", BCIR_CAP_DATA_PARALLEL},
    {"reduce", BCIR_CAP_REDUCE}, {"gather", BCIR_CAP_GATHER}, {"tile", BCIR_CAP_TILE},
    {"matmul", BCIR_CAP_MATMUL}, {"stream_unit", BCIR_CAP_STREAM_UNIT},
    {"scalar_stream", BCIR_CAP_SCALAR_STREAM}, {NULL, 0}};
  for (int i = 0; C[i].n; i++) if (!strcmp(C[i].n, s)) return C[i].b;
  return 0;
}

static int valid_kind(const char *s) {
  return !strcmp(s, "cpu") || !strcmp(s, "gpu") || !strcmp(s, "fpga") ||
         !strcmp(s, "accelerator") || !strcmp(s, "storage") || !strcmp(s, "memory");
}

static int valid_provenance(const char *s) {
  return !strcmp(s, "real") || !strcmp(s, "modeled") || !strcmp(s, "simulated");
}

static int valid_text(const char *s) {
  for (; *s; s++) if ((unsigned char)*s < 0x20u) return 0;
  return 1;
}

int bcir_channel_parse(const char *json, size_t len, bcir_channel *ch, char *diag, size_t dn) {
  if (!diag) dn = 0;
  if (dn) diag[0] = 0;
  if (ch) memset(ch, 0, sizeof *ch);
  if (!json || !ch) { if (dn) snprintf(diag, dn, "channel.json: invalid arguments"); return 1; }
  if (len > BCIR_CHANNEL_JSON_MAX_BYTES) {
    if (dn) snprintf(diag, dn, "channel.json: input exceeds %u bytes",
                     (unsigned)BCIR_CHANNEL_JSON_MAX_BYTES);
    return 1;
  }
  bcir_channel tmp;
  memset(&tmp, 0, sizeof tmp); tmp.modeled = 1;
  char seen_keys[64][48];
  size_t nkeys = 0;
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
    for (size_t i = 0; i < nkeys; i++) {
      if (!strcmp(seen_keys[i], key)) {
        snprintf(diag, dn, "channel.json: duplicate key '%s'", key); return 1;
      }
    }
    if (nkeys == sizeof seen_keys / sizeof seen_keys[0]) {
      snprintf(diag, dn, "channel.json: too many top-level fields"); return 1;
    }
    memcpy(seen_keys[nkeys], key, strlen(key) + 1u); nkeys++;
    p = jws(p, e);
    if (p >= e || *p != ':') { snprintf(diag, dn, "channel.json: expected ':'"); return 1; }
    p++; p = jws(p, e);
    if (!strcmp(key, "name")) p = jstr(p, e, tmp.name, sizeof tmp.name);
    else if (!strcmp(key, "kind")) p = jstr(p, e, tmp.kind, sizeof tmp.kind);
    else if (!strcmp(key, "provenance")) p = jstr(p, e, tmp.provenance, sizeof tmp.provenance);
    else if (!strcmp(key, "modeled")) {
      if ((size_t)(e-p)>=4u&&!strncmp(p,"true",4)){tmp.modeled=1;p+=4;}
      else if ((size_t)(e-p)>=5u&&!strncmp(p,"false",5)){tmp.modeled=0;p+=5;}
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
          uint32_t bit = cap_bit(cap);
          if (!bit) { snprintf(diag, dn, "channel.json: unknown capability '%s'", cap); return 1; }
          if (tmp.capabilities & bit) {
            snprintf(diag, dn, "channel.json: duplicate capability '%s'", cap); return 1;
          }
          tmp.capabilities |= bit;
          p = jws(p, e);
          if (p < e && *p == ',') { p++; p=jws(p,e); if(p>=e||*p==']'){snprintf(diag,dn,"channel.json: trailing capability comma");return 1;} }
          else if (p >= e || *p != ']') { snprintf(diag,dn,"channel.json: expected ',' or ']'");return 1; }
        }
      } else { snprintf(diag,dn,"channel.json: 'capabilities' must be an array");return 1; }
    } else p = jskip(p, e);
    if (!p) { snprintf(diag, dn, "channel.json: parse error"); return 1; }
    p = jws(p, e);
    if (p < e && *p == ',') { p++; p=jws(p,e); if(p>=e||*p=='}'){snprintf(diag,dn,"channel.json: trailing comma");return 1;} }
    else if (p >= e || *p != '}') { snprintf(diag,dn,"channel.json: expected ',' or '}'");return 1; }
  }
  if (jws(p,e) != e) { snprintf(diag,dn,"channel.json: trailing data");return 1; }
  if (!tmp.name[0]) { snprintf(diag, dn, "channel.json: missing 'name'"); return 1; }
  if (!valid_text(tmp.name)) {
    snprintf(diag, dn, "channel.json: 'name' contains a control character"); return 1;
  }
  if (!tmp.kind[0]) { snprintf(diag, dn, "channel.json: missing 'kind'"); return 1; }
  if (!valid_kind(tmp.kind)) { snprintf(diag, dn, "channel.json: unknown kind '%s'", tmp.kind); return 1; }
  if (tmp.provenance[0] && !valid_provenance(tmp.provenance)) {
    snprintf(diag, dn, "channel.json: unknown provenance '%s'", tmp.provenance); return 1;
  }
  if (!strcmp(tmp.provenance, "real") && tmp.modeled) {
    snprintf(diag, dn, "channel.json: real provenance contradicts modeled=true"); return 1;
  }
  *ch = tmp;
  return 0;
}

/* --- the routing decision (mirror bcir/channels: claim_required_caps / channel_suits / route) --- */
static int starts_with(const char *s, const char *pre) {
  size_t n = strlen(pre); return strncmp(s, pre, n) == 0;
}

static int claim_routing_valid(const bcir_claim *cl) {
  return cl && memchr(cl->op, 0, sizeof cl->op) != NULL;
}

static int channel_routing_valid(const bcir_channel *ch) {
  const uint32_t all_caps = BCIR_CAP_UNIVERSAL | BCIR_CAP_DATA_PARALLEL | BCIR_CAP_REDUCE |
      BCIR_CAP_GATHER | BCIR_CAP_TILE | BCIR_CAP_MATMUL | BCIR_CAP_STREAM_UNIT |
      BCIR_CAP_SCALAR_STREAM;
  return ch && memchr(ch->name, 0, sizeof ch->name) != NULL &&
      memchr(ch->kind, 0, sizeof ch->kind) != NULL && (ch->capabilities & ~all_caps) == 0;
}

uint32_t bcir_claim_required_caps(const bcir_claim *cl) {
  if (!claim_routing_valid(cl)) return 0;
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
  if (!claim_routing_valid(cl) || !channel_routing_valid(ch)) return 0;
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
  if (!claim_routing_valid(cl) || n <= 0 || !channels) return -1;
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
