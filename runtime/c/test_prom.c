/*===- test_prom.c - the Prometheus exposition: contracts, and a driver for the oracle's comparison ---------===
 *
 *   test_prom          the contracts the oracle cannot state: a short output says the text's length and writes
 *                      nothing, names out of order or not UTF-8 refused, a reading without its definition, every
 *                      NULL, and a known snapshot's text. Prints `test_prom: OK`.
 *   test_prom FILE     one request per line, as bcir/tests/test_c_prom.py writes them:
 *                        `prom CHANNEL N SERIES... K SENS... WITNESS`   -> `ok HEX` | `ERR CODE`
 *                      a SERIES `NAME AVAIL HASDEF COUNTER DEFNAME DESC COST PROV UNIT VALUE`, a SENS `SIGNAL ABS`,
 *                      the WITNESS `-` or its eight counters; texts in hexadecimal, `-` for empty.
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_prom.h"

static int failures;
#define CHECK(cond)                                                                                  \
  do {                                                                                               \
    if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; }          \
  } while (0)

static bcir_prom_text T(const char *s) {
  bcir_prom_text t = {(const uint8_t *)s, strlen(s)};
  return t;
}

static const char known[] =
    "# HELP bcir_signal_up 1 if the telemetry source is available on this host, else 0 (honest real/unavailable "
    "split).\n"
    "# TYPE bcir_signal_up gauge\n"
    "bcir_signal_up{channel=\"h\\\"1\",cost_dim=\"thermal\",provenance=\"measured\",unit=\"millicelsius\"} 1\n"
    "bcir_signal_up{channel=\"h\\\"1\"} 0\n"
    "# HELP bcir_thermal_die die temp\n"
    "# TYPE bcir_thermal_die gauge\n"
    "bcir_thermal_die{channel=\"h\\\"1\",cost_dim=\"thermal\",provenance=\"measured\",unit=\"millicelsius\"} -5\n";

static void contracts(void) {
  bcir_prom_series s[2];
  memset(s, 0, sizeof s);
  s[0].name = T("thermal.die");
  s[0].available = s[0].has_definition = 1;
  s[0].definition_name = T("thermal.die");
  s[0].description = T("  die\ntemp\n ");
  s[0].cost_dim = T("thermal");
  s[0].provenance = T("measured");
  s[0].unit = T("millicelsius");
  s[0].value = -5;
  s[1].name = T("zz.absent");
  static uint8_t out[4096];
  size_t len = 0;
  CHECK(bcir_prom_write(s, 2, T("h\"1"), NULL, 0, NULL, out, sizeof out, &len) == BCIR_PROM_OK);
  CHECK(len == strlen(known) && memcmp(out, known, len) == 0);
  /* a short output: the length it needs, nothing written */
  memset(out, 0xEE, sizeof out);
  CHECK(bcir_prom_write(s, 2, T("h\"1"), NULL, 0, NULL, out, len - 1u, &len) == BCIR_PROM_ERR_SPACE &&
        len == strlen(known) && out[0] == 0xEE);
  CHECK(bcir_prom_write(s, 2, T("h\"1"), NULL, 0, NULL, NULL, 0, &len) == BCIR_PROM_ERR_SPACE);
  /* the order, UTF-8, a reading without its definition */
  bcir_prom_series r[2] = {s[1], s[0]};
  CHECK(bcir_prom_write(r, 2, T("h"), NULL, 0, NULL, out, sizeof out, &len) == BCIR_PROM_ERR_ORDER);
  r[0] = s[0];
  r[1] = s[0];
  CHECK(bcir_prom_write(r, 2, T("h"), NULL, 0, NULL, out, sizeof out, &len) == BCIR_PROM_ERR_ORDER);
  r[1] = s[1];
  r[1].name = T("zz\xC0\xAF"); /* an overlong '/' */
  CHECK(bcir_prom_write(r, 2, T("h"), NULL, 0, NULL, out, sizeof out, &len) == BCIR_PROM_ERR_UTF8);
  r[1].name = T("zz\xED\xA0\x80"); /* a surrogate */
  CHECK(bcir_prom_write(r, 2, T("h"), NULL, 0, NULL, out, sizeof out, &len) == BCIR_PROM_ERR_UTF8);
  CHECK(bcir_prom_write(s, 2, T("\xFF"), NULL, 0, NULL, out, sizeof out, &len) == BCIR_PROM_ERR_UTF8);
  r[1] = s[1];
  r[1].available = 1;
  CHECK(bcir_prom_write(r, 2, T("h"), NULL, 0, NULL, out, sizeof out, &len) == BCIR_PROM_ERR_ARG);
  /* a NULL name of a nonzero length, after a name and before one: refused, never read */
  r[0] = s[0];
  r[1] = s[1];
  r[1].name.p = NULL;
  CHECK(bcir_prom_write(r, 2, T("h"), NULL, 0, NULL, out, sizeof out, &len) == BCIR_PROM_ERR_ARG);
  r[1] = s[1];
  r[0].name.p = NULL;
  CHECK(bcir_prom_write(r, 2, T("h"), NULL, 0, NULL, out, sizeof out, &len) == BCIR_PROM_ERR_ARG);
  /* every NULL */
  CHECK(bcir_prom_write(NULL, 2, T("h"), NULL, 0, NULL, out, sizeof out, &len) == BCIR_PROM_ERR_ARG);
  CHECK(bcir_prom_write(s, 2, T("h"), NULL, 1, NULL, out, sizeof out, &len) == BCIR_PROM_ERR_ARG);
  CHECK(bcir_prom_write(s, 2, T("h"), NULL, 0, NULL, out, sizeof out, NULL) == BCIR_PROM_ERR_ARG);
  CHECK(bcir_prom_write(s, 2, T("h"), NULL, 0, NULL, NULL, 8, &len) == BCIR_PROM_ERR_ARG);
  /* counters past 2**63 print unsigned */
  bcir_prom_witness wit = {UINT64_MAX, 0, 0, 0, 0, 0, 0, 0};
  CHECK(bcir_prom_write(NULL, 0, T("h"), NULL, 0, &wit, out, sizeof out, &len) == BCIR_PROM_OK);
  out[len] = 0;
  CHECK(strstr((const char *)out, "bcir_telemetry_accepted{channel=\"h\"} 18446744073709551615\n") != NULL);
  CHECK(strstr((const char *)out, "frames") == NULL && strstr((const char *)out, "bcir_telemetry_blind{channel=\"h\"} 0"));
}

static uint8_t pool[1 << 22];
static size_t pool_used;

static int unhex(char **p, bcir_prom_text *t) {
  char *s = *p;
  while (*s == ' ') s++;
  char *e = s;
  while (*e && *e != ' ' && *e != '\n') e++;
  size_t n = (size_t)(e - s);
  *p = e;
  t->p = pool + pool_used;
  t->n = 0;
  if (n == 1 && *s == '-') return 1;
  if (n % 2u || pool_used + n / 2u > sizeof pool) return 0;
  for (size_t k = 0; k < n; k += 2) {
    unsigned v;
    if (sscanf(s + k, "%2x", &v) != 1) return 0;
    pool[pool_used++] = (uint8_t)v;
  }
  t->n = n / 2u;
  return 1;
}

static int num(char **p, long long *v) {
  char *end;
  *v = strtoll(*p, &end, 10);
  if (end == *p) return 0;
  *p = end;
  return 1;
}

static int unum(char **p, unsigned long long *v) {
  char *end;
  *v = strtoull(*p, &end, 10);
  if (end == *p) return 0;
  *p = end;
  return 1;
}

static bcir_prom_series series[4096];
static bcir_prom_sensitivity sens[1024];
static uint8_t out[1 << 22];

static int request(char *p) {
  bcir_prom_text channel;
  long long n, k, v;
  if (!unhex(&p, &channel) || !num(&p, &n) || n < 0 || n > 4096) return 0;
  for (long long i = 0; i < n; i++) {
    bcir_prom_series *s = &series[i];
    long long a, h, c;
    if (!unhex(&p, &s->name) || !num(&p, &a) || !num(&p, &h) || !num(&p, &c) || !unhex(&p, &s->definition_name) ||
        !unhex(&p, &s->description) || !unhex(&p, &s->cost_dim) || !unhex(&p, &s->provenance) ||
        !unhex(&p, &s->unit) || !num(&p, &v))
      return 0;
    s->available = (int)a;
    s->has_definition = (int)h;
    s->counter = (int)c;
    s->value = (int64_t)v;
  }
  if (!num(&p, &k) || k < 0 || k > 1024) return 0;
  for (long long i = 0; i < k; i++)
    if (!unhex(&p, &sens[i].signal) || !num(&p, &v)) return 0;
    else sens[i].abs_delta = (int64_t)v;
  while (*p == ' ') p++;
  bcir_prom_witness wit, *w = NULL;
  if (*p != '-') {
    unsigned long long c[8];
    for (int j = 0; j < 8; j++)
      if (!unum(&p, &c[j])) return 0;
    wit.accepted = c[0], wit.rejected = c[1], wit.dropped = c[2], wit.frames_accepted = c[3];
    wit.frames_rejected = c[4], wit.frames_missing = c[5], wit.frames_reordered = c[6], wit.frames_duplicated = c[7];
    w = &wit;
  }
  size_t len = 0;
  int r = bcir_prom_write(series, (size_t)n, channel, sens, (size_t)k, w, out, sizeof out, &len);
  if (r) {
    printf("ERR %d\n", r);
    return 1;
  }
  printf("ok ");
  for (size_t j = 0; j < len; j++) printf("%02x", out[j]);
  printf("\n");
  return 1;
}

static int drive(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) { printf("TOOL cannot read the file\n"); return 2; }
  static char line[1 << 22];
  int rc = 0;
  while (fgets(line, sizeof line, f)) {
    pool_used = 0;
    if (strncmp(line, "prom ", 5) || !request(line + 5)) {
      rc = 2;
      break;
    }
  }
  if (rc) printf("TOOL malformed line\n");
  fclose(f);
  return rc;
}

int main(int argc, char **argv) {
  if (argc == 2) return drive(argv[1]);
  contracts();
  if (failures) { printf("test_prom: %d FAILED\n", failures); return 1; }
  printf("test_prom: OK\n");
  return 0;
}
