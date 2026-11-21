/*===- bcir_prom.c - the telemetry export's Prometheus text exposition --------------------------------------===
 *
 * See bcir_prom.h. The text is measured first and written only when it fits, so a short output is untouched.
 *===----------------------------------------------------------------------===*/
#include "bcir_prom.h"

typedef struct writer {
  uint8_t *out; /* NULL while measuring */
  size_t len;
} writer;

static void raw(writer *w, const uint8_t *p, size_t n) {
  if (w->out)
    for (size_t k = 0; k < n; k++) w->out[w->len + k] = p[k];
  w->len += n;
}

static void lit(writer *w, const char *s) {
  size_t n = 0;
  while (s[n]) n++;
  raw(w, (const uint8_t *)s, n);
}

static void decimal(writer *w, uint64_t u, int negative) {
  uint8_t buf[24];
  size_t n = sizeof buf;
  do {
    buf[--n] = (uint8_t)('0' + u % 10u);
    u /= 10u;
  } while (u);
  if (negative) buf[--n] = '-';
  raw(w, buf + n, sizeof buf - n);
}

/* str(int) */
static void int64(writer *w, int64_t v) { decimal(w, v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v, v < 0); }

/* one UTF-8 scalar at p[k]: its width, or 0 when the octets are not UTF-8 (overlong, surrogate, past U+10FFFF) */
static size_t utf8_next(const uint8_t *p, size_t n, size_t k, uint32_t *cp) {
  uint8_t c = p[k];
  size_t w;
  uint32_t v, min;
  if (c < 0x80u) {
    *cp = c;
    return 1;
  } else if ((c & 0xE0u) == 0xC0u) {
    w = 2;
    v = c & 0x1Fu;
    min = 0x80u;
  } else if ((c & 0xF0u) == 0xE0u) {
    w = 3;
    v = c & 0x0Fu;
    min = 0x800u;
  } else if ((c & 0xF8u) == 0xF0u) {
    w = 4;
    v = c & 0x07u;
    min = 0x10000u;
  } else {
    return 0;
  }
  if (n - k < w) return 0;
  for (size_t j = 1; j < w; j++) {
    if ((p[k + j] & 0xC0u) != 0x80u) return 0;
    v = (v << 6) | (p[k + j] & 0x3Fu);
  }
  if (v < min || v > 0x10FFFFu || (v >= 0xD800u && v <= 0xDFFFu)) return 0;
  *cp = v;
  return w;
}

static int utf8_ok(bcir_prom_text t) {
  uint32_t cp;
  for (size_t k = 0; k < t.n;) {
    size_t w = utf8_next(t.p, t.n, k, &cp);
    if (!w) return 0;
    k += w;
  }
  return 1;
}

static int name_char(uint32_t c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == ':';
}

/* sanitize_metric_name: "bcir_", then the name with every code point outside [a-zA-Z0-9_:] as "_", led by "_"
 * when it would start with a digit */
static void metric_name(writer *w, bcir_prom_text name) {
  lit(w, "bcir_");
  if (name.n && name.p[0] >= '0' && name.p[0] <= '9') lit(w, "_");
  for (size_t k = 0; k < name.n;) {
    uint32_t cp = 0;
    size_t step = utf8_next(name.p, name.n, k, &cp);
    if (name_char(cp)) raw(w, name.p + k, 1);
    else lit(w, "_");
    k += step;
  }
}

/* a label value: a backslash, a quote and a newline escaped */
static void label_value(writer *w, bcir_prom_text v) {
  for (size_t k = 0; k < v.n; k++) {
    uint8_t c = v.p[k];
    if (c == '\\') lit(w, "\\\\");
    else if (c == '"') lit(w, "\\\"");
    else if (c == '\n') lit(w, "\\n");
    else raw(w, &c, 1);
  }
}

/* str.isspace(): Python's whitespace, which str.strip() removes from both ends */
static int py_space(uint32_t c) {
  return (c >= 0x09u && c <= 0x0Du) || (c >= 0x1Cu && c <= 0x20u) || c == 0x85u || c == 0xA0u || c == 0x1680u ||
         (c >= 0x2000u && c <= 0x200Au) || c == 0x2028u || c == 0x2029u || c == 0x202Fu || c == 0x205Fu ||
         c == 0x3000u;
}

/* _help_text: newlines folded to spaces, then Python's whitespace stripped from both ends */
static void help_text(writer *w, bcir_prom_text d) {
  size_t lo = 0, hi = d.n; /* [lo, hi): the text between the stripped ends, in octets */
  uint32_t cp = 0;
  while (lo < hi) {
    size_t step = utf8_next(d.p, d.n, lo, &cp);
    if (!py_space(cp)) break;
    lo += step;
  }
  size_t last = lo; /* the end of the last code point that is not whitespace */
  for (size_t k = lo; k < hi;) {
    size_t step = utf8_next(d.p, d.n, k, &cp);
    k += step;
    if (!py_space(cp)) last = k;
  }
  for (size_t k = lo; k < last; k++) {
    if (d.p[k] == '\n') lit(w, " ");
    else raw(w, d.p + k, 1);
  }
}

static bcir_prom_text t_of(const char *s) {
  bcir_prom_text t;
  size_t n = 0;
  while (s[n]) n++;
  t.p = (const uint8_t *)s;
  t.n = n;
  return t;
}

/* {channel, cost_dim, provenance, unit}, or {channel} for a signal with no known definition */
static void labels(writer *w, const bcir_prom_series *s, bcir_prom_text channel) {
  lit(w, "{channel=\"");
  label_value(w, channel);
  if (s->has_definition) {
    lit(w, "\",cost_dim=\"");
    label_value(w, s->cost_dim.n ? s->cost_dim : t_of("none"));
    lit(w, "\",provenance=\"");
    label_value(w, s->provenance);
    lit(w, "\",unit=\"");
    label_value(w, s->unit);
  }
  lit(w, "\"}");
}

static void counter(writer *w, const char *suffix, const char *help, uint64_t v, bcir_prom_text channel) {
  lit(w, "# HELP bcir_");
  lit(w, suffix);
  lit(w, " ");
  lit(w, help);
  lit(w, "\n# TYPE bcir_");
  lit(w, suffix);
  lit(w, " counter\nbcir_");
  lit(w, suffix);
  lit(w, "{channel=\"");
  label_value(w, channel);
  lit(w, "\"} ");
  decimal(w, v, 0);
  lit(w, "\n");
}

static void render(writer *w, const bcir_prom_series *series, size_t n, bcir_prom_text channel,
                   const bcir_prom_sensitivity *sens, size_t n_sens, const bcir_prom_witness *wit) {
  lit(w, "# HELP bcir_signal_up 1 if the telemetry source is available on this host, else 0 (honest "
         "real/unavailable split).\n# TYPE bcir_signal_up gauge\n");
  for (size_t k = 0; k < n; k++) {
    lit(w, "bcir_signal_up");
    labels(w, &series[k], channel);
    lit(w, series[k].available ? " 1\n" : " 0\n");
  }
  for (size_t k = 0; k < n; k++) {
    const bcir_prom_series *s = &series[k];
    if (!s->available) continue; /* no value line for a signal that did not read: honest */
    lit(w, "# HELP ");
    metric_name(w, s->name);
    lit(w, " ");
    help_text(w, s->description.n ? s->description : s->definition_name);
    lit(w, "\n# TYPE ");
    metric_name(w, s->name);
    lit(w, s->counter ? " counter\n" : " gauge\n");
    metric_name(w, s->name);
    labels(w, s, channel);
    lit(w, " ");
    int64(w, s->value);
    lit(w, "\n");
  }
  if (n_sens) {
    lit(w, "# HELP bcir_signal_sensitivity T3 .SENS |Delta score| of a telemetry signal on the optimized plan cost "
           "(sampling-budget rank).\n# TYPE bcir_signal_sensitivity gauge\n");
    for (size_t k = 0; k < n_sens; k++) {
      lit(w, "bcir_signal_sensitivity{signal=\"");
      label_value(w, sens[k].signal);
      lit(w, "\"} ");
      int64(w, sens[k].abs_delta);
      lit(w, "\n");
    }
  }
  if (!wit) return;
  counter(w, "telemetry_accepted", "Telemetry records that passed the documented RT3 schema (the stream up-count).",
          wit->accepted, channel);
  counter(w, "telemetry_rejected", "Telemetry records dropped for a schema violation (value injection).",
          wit->rejected, channel);
  counter(w, "telemetry_dropped", "Telemetry records evicted upstream before read (ring overwrite / suppression).",
          wit->dropped, channel);
  int frames = wit->frames_accepted || wit->frames_rejected || wit->frames_missing || wit->frames_reordered ||
               wit->frames_duplicated;
  if (frames) {
    counter(w, "telemetry_frames_accepted", "BTLM telemetry frames that passed flags/version/length/CRC validation.",
            wit->frames_accepted, channel);
    counter(w, "telemetry_frames_rejected",
            "Magic-aligned telemetry candidates rejected for flags/version/length/CRC.", wit->frames_rejected, channel);
    counter(w, "telemetry_frames_missing", "Telemetry frame sequence numbers missing between recovered frames.",
            wit->frames_missing, channel);
    counter(w, "telemetry_frames_reordered", "Telemetry frames received behind the sequence watermark.",
            wit->frames_reordered, channel);
    counter(w, "telemetry_frames_duplicated", "Telemetry frames repeating the current sequence number.",
            wit->frames_duplicated, channel);
    lit(w, "# HELP bcir_telemetry_frame_monotonic 1 if recovered frame sequence order has no duplicate/reorder, else "
           "0.\n# TYPE bcir_telemetry_frame_monotonic gauge\nbcir_telemetry_frame_monotonic{channel=\"");
    label_value(w, channel);
    lit(w, wit->frames_reordered == 0 && wit->frames_duplicated == 0 ? "\"} 1\n" : "\"} 0\n");
  }
  lit(w, "# HELP bcir_telemetry_blind 1 if no valid telemetry survived (suppression / all-rejected / all-dropped), "
         "else 0.\n# TYPE bcir_telemetry_blind gauge\nbcir_telemetry_blind{channel=\"");
  label_value(w, channel);
  lit(w, wit->accepted == 0 ? "\"} 1\n" : "\"} 0\n");
}

static int bytes_lt(bcir_prom_text a, bcir_prom_text b) {
  size_t n = a.n < b.n ? a.n : b.n;
  for (size_t k = 0; k < n; k++)
    if (a.p[k] != b.p[k]) return a.p[k] < b.p[k];
  return a.n < b.n;
}

static int text_ok(bcir_prom_text t) { return (t.p || !t.n) && utf8_ok(t); }

int bcir_prom_write(const bcir_prom_series *series, size_t n, bcir_prom_text channel,
                    const bcir_prom_sensitivity *sens, size_t n_sens, const bcir_prom_witness *witness,
                    uint8_t *out, size_t cap, size_t *len) {
  if (len) *len = 0;
  if (!len || (!series && n) || (!sens && n_sens) || (!out && cap)) return BCIR_PROM_ERR_ARG;
  int utf8 = text_ok(channel);
  for (size_t k = 0; k < n; k++) {
    const bcir_prom_series *s = &series[k];
    if (s->available && !s->has_definition) return BCIR_PROM_ERR_ARG;
    if (k && !bytes_lt(series[k - 1u].name, s->name)) return BCIR_PROM_ERR_ORDER;
    utf8 = utf8 && text_ok(s->name);
    if (s->has_definition)
      utf8 = utf8 && text_ok(s->definition_name) && text_ok(s->description) && text_ok(s->cost_dim) &&
             text_ok(s->provenance) && text_ok(s->unit);
  }
  for (size_t k = 0; k < n_sens; k++) utf8 = utf8 && text_ok(sens[k].signal);
  if (!utf8) return BCIR_PROM_ERR_UTF8;
  writer w = {0, 0};
  render(&w, series, n, channel, sens, n_sens, witness);
  *len = w.len;
  if (w.len > cap) return BCIR_PROM_ERR_SPACE;
  w.out = out;
  w.len = 0;
  render(&w, series, n, channel, sens, n_sens, witness);
  return BCIR_PROM_OK;
}
