/* BCIR native cost microbench (the measurement half of the L1 calibration loop).
 * Times four access regimes over a cache-defeating buffer -- streaming, strided,
 * random (gather), pure compute -- and prints a frozen Q8 cost table as JSON
 * matching bcir.kbcir.microbench.CalibratedProfile, plus the EVIDENCE the table
 * rests on (S0-F / GEM+ G7):
 *
 *   - a full-cycle strided walk: `(k * stride) % n` with a power-of-two n visits only
 *     n / gcd(n, stride) elements (n/16 for the default: a 2 MiB working set under a
 *     nominal 32 MiB buffer). The walk now runs the gcd(stride, n) cosets of <stride>
 *     in Z_n, each a full cycle, so every element is touched exactly once -- and a
 *     non-timed census COUNTS the unique elements per regime rather than assuming them;
 *   - the raw per-regime samples (ns) with min / median / max / MAD and the outlier
 *     policy (median of repeats, nothing discarded), so a reader can judge dispersion
 *     instead of trusting a ratio;
 *   - an attestation of the environment read from /proc and /sys: hypervisor flag, DMI
 *     vendor/product, WSL, container, hypervisor nodes, the hardware PMU event source,
 *     perf_event_paranoid, cpufreq governor, RAPL, clocksource, the observed timer
 *     quantum, OS/arch/compiler -- and a TENANCY derived from that evidence.
 *     "bare-metal" is reserved for a host with no virtualization or container signal
 *     AND an exposed hardware PMU; a hypervisor/WSL signal is "virtualized", a container
 *     signal alone is "containerized", anything else is "unproven". The provenance
 *     string carries the verdict and its evidence; the old rig printed "bare-metal"
 *     unconditionally, under WSL and under this hypervisor alike.
 *
 * Deterministic access orders (a seeded LCG permutation; only the timing varies).
 * This is offline (L2/L3) tooling -- never the hot path; the planner consumes only the
 * frozen table it prints. The Python reader (CalibratedProfile.from_json) re-derives
 * the Q8 ratios from the medians of the samples it is handed and refuses a table whose
 * summary disagrees with its own evidence.
 *
 *   cc -O2 -std=c11 bcir_microbench.c -o mb && ./mb [n] [repeats] [cal_gen]
 */
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bcir_host_alloc.h"

#define Q8 256
#define BCIR_MICROBENCH_MAX_N (UINT64_C(1) << 24)
#define BCIR_MICROBENCH_MAX_REPEATS UINT64_C(128)
#define BCIR_MICROBENCH_STRIDE ((size_t)16)
#define BCIR_MICROBENCH_LINE 256

static uint64_t g_state = 0xBC12u;
static uint64_t lcg(void) {
  g_state = g_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return g_state;
}

static uint64_t now_ns(void) {
  /* ISO C11 timespec_get (no POSIX feature macros needed; portable under -std=c11). */
  struct timespec t = {0, 0};
  if (timespec_get(&t, TIME_UTC) != TIME_UTC || t.tv_sec < 0 || t.tv_nsec < 0)
    return 0;
  uint64_t sec = (uint64_t)t.tv_sec, ns = (uint64_t)t.tv_nsec;
  if (sec > (UINT64_MAX - ns) / UINT64_C(1000000000)) return UINT64_MAX;
  return sec * UINT64_C(1000000000) + ns;
}

static int cmp_u64(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return (x > y) - (x < y);
}

/* min / median / max / median-absolute-deviation of k samples (sorts a scratch copy). */
typedef struct { uint64_t min, median, max, mad; } stat4;

static stat4 summarize(const uint64_t *xs, int k, uint64_t *scratch) {
  stat4 s = {0, 0, 0, 0};
  int i;
  for (i = 0; i < k; ++i) scratch[i] = xs[i];
  qsort(scratch, (size_t)k, sizeof(uint64_t), cmp_u64);
  s.min = scratch[0]; s.median = scratch[k / 2]; s.max = scratch[k - 1];
  for (i = 0; i < k; ++i)
    scratch[i] = xs[i] > s.median ? xs[i] - s.median : s.median - xs[i];
  qsort(scratch, (size_t)k, sizeof(uint64_t), cmp_u64);
  s.mad = scratch[k / 2];
  return s;
}

static size_t gcd_size(size_t a, size_t b) {
  while (b) { size_t t = a % b; a = b; b = t; }
  return a;
}

/* one pass touching n indices; volatile sink keeps the loop live under -O2. */
static volatile double g_sink = 0.0;

static uint64_t elapsed_since(uint64_t start) {
  uint64_t end = now_ns();
  return end > start ? end - start : 1;
}

static uint64_t pass_stream(const double *buf, size_t n) {
  uint64_t t0 = now_ns();
  double s = 0.0;
  for (size_t k = 0; k < n; ++k) s += buf[k];
  uint64_t dt = elapsed_since(t0);
  g_sink += s;
  return dt;
}

/* The full-cycle strided walk: the g = gcd(stride, n) cosets {c, c+stride, c+2*stride,
 * ...} of <stride> in Z_n partition Z_n, each of length n/g, so every element is
 * visited exactly once at the declared stride. No modulo in the timed loop: the step
 * is reduced once and the index wraps by one conditional subtraction. */
static uint64_t pass_strided(const double *buf, size_t n, size_t stride) {
  size_t step = stride % n, g = gcd_size(stride, n), cycle = n / g;
  uint64_t t0 = now_ns();
  double s = 0.0;
  for (size_t c = 0; c < g; ++c) {
    size_t idx = c;
    for (size_t k = 0; k < cycle; ++k) {
      s += buf[idx];
      idx += step;
      if (idx >= n) idx -= n;
    }
  }
  uint64_t dt = elapsed_since(t0);
  g_sink += s;
  return dt;
}

static uint64_t pass_random(const double *buf, const size_t *perm, size_t n) {
  uint64_t t0 = now_ns();
  double s = 0.0;
  for (size_t k = 0; k < n; ++k) s += buf[perm[k]];
  uint64_t dt = elapsed_since(t0);
  g_sink += s;
  return dt;
}

static uint64_t pass_compute(size_t n) {
  uint64_t t0 = now_ns();
  double s = 1.0;
  for (size_t k = 0; k < n; ++k) s = s * 1.0000001 + 0.5;
  uint64_t dt = elapsed_since(t0);
  g_sink += s;
  return dt;
}

/* --- the census: count the unique elements a regime touches (never assumed) ------- */

static size_t census_mark(unsigned char *seen, size_t n, size_t idx) {
  if (idx >= n) return 0;
  if (seen[idx >> 3] & (unsigned char)(1u << (idx & 7u))) return 0;
  seen[idx >> 3] |= (unsigned char)(1u << (idx & 7u));
  return 1;
}

static void census_clear(unsigned char *seen, size_t n) {
  size_t bytes = (n + 7u) / 8u, i;
  for (i = 0; i < bytes; ++i) seen[i] = 0;
}

static size_t census_strided(unsigned char *seen, size_t n, size_t stride) {
  size_t step = stride % n, g = gcd_size(stride, n), cycle = n / g, unique = 0;
  census_clear(seen, n);
  for (size_t c = 0; c < g; ++c) {
    size_t idx = c;
    for (size_t k = 0; k < cycle; ++k) {
      unique += census_mark(seen, n, idx);
      idx += step;
      if (idx >= n) idx -= n;
    }
  }
  return unique;
}

static size_t census_random(unsigned char *seen, const size_t *perm, size_t n) {
  size_t unique = 0;
  census_clear(seen, n);
  for (size_t k = 0; k < n; ++k) unique += census_mark(seen, n, perm[k]);
  return unique;
}

/* --- the environment attestation (read-only probes of /proc and /sys) ------------ */

static int file_exists(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  fclose(f);
  return 1;
}

/* First line of `path` (trailing newline stripped) into `out`, or "unavailable". */
static void file_first_line(const char *path, char *out, size_t cap) {
  FILE *f = fopen(path, "r");
  size_t len;
  if (cap == 0) return;
  out[0] = 0;
  if (!f || !fgets(out, (int)cap, f)) {
    if (f) fclose(f);
    strncpy(out, "unavailable", cap - 1);
    out[cap - 1] = 0;
    return;
  }
  fclose(f);
  len = strlen(out);
  while (len && (out[len - 1] == '\n' || out[len - 1] == '\r')) out[--len] = 0;
  if (!len) { strncpy(out, "unavailable", cap - 1); out[cap - 1] = 0; }
}

static void lower_ascii(char *s) {
  for (; *s; ++s) *s = (char)tolower((unsigned char)*s);
}

/* Whether any line of `path` contains `needle` (both compared lower-cased). */
static int file_contains(const char *path, const char *needle) {
  char line[BCIR_MICROBENCH_LINE];
  FILE *f = fopen(path, "r");
  int found = 0;
  if (!f) return 0;
  while (!found && fgets(line, (int)sizeof line, f)) {
    lower_ascii(line);
    if (strstr(line, needle)) found = 1;
  }
  fclose(f);
  return found;
}

static long file_long(const char *path, long unavailable) {
  char line[BCIR_MICROBENCH_LINE];
  char *end;
  long v;
  file_first_line(path, line, sizeof line);
  if (!strcmp(line, "unavailable")) return unavailable;
  errno = 0;
  v = strtol(line, &end, 10);
  if (errno || end == line) return unavailable;
  return v;
}

typedef struct {
  int hypervisor_flag, wsl, container, hypervisor_node, dmi_virtual;
  char dmi_vendor[BCIR_MICROBENCH_LINE], dmi_product[BCIR_MICROBENCH_LINE];
  int hardware_pmu;
  const char *pmu_source;
  long perf_event_paranoid;
  char governor[BCIR_MICROBENCH_LINE], clocksource[BCIR_MICROBENCH_LINE];
  int rapl;
  uint64_t timer_quantum_ns;
  const char *tenancy;
  char signals[BCIR_MICROBENCH_LINE];
} attestation;

static const char *const k_dmi_virtual[] = {
    "kvm", "qemu", "vmware", "virtualbox", "innotek", "xen", "bochs", "parallels",
    "hyper-v", "virtual machine", "amazon ec2", "google compute engine", "openstack",
    "bhyve", "cloud hypervisor", "firecracker", 0};

static const char *const k_pmu_sources[] = {"cpu", "cpu_core", "cpu_atom", "armv8_pmuv3",
                                             "armv8_pmuv3_0", 0};

static void append_signal(char *signals, size_t cap, const char *signal) {
  size_t len = strlen(signals);
  if (len && len + 1 < cap) { signals[len++] = ','; signals[len] = 0; }
  strncat(signals, signal, cap - strlen(signals) - 1);
}

static uint64_t observe_timer_quantum(void) {
  uint64_t best = 0;
  int i;
  for (i = 0; i < 20000; ++i) {
    uint64_t a = now_ns(), b = now_ns();
    if (b > a && (best == 0 || b - a < best)) best = b - a;
  }
  return best;
}

static void attest(attestation *a) {
  char lowered[BCIR_MICROBENCH_LINE];
  int i;
  memset(a, 0, sizeof *a);
  a->hypervisor_flag = file_contains("/proc/cpuinfo", " hypervisor");
  a->wsl = file_contains("/proc/version", "microsoft") ||
           file_exists("/proc/sys/fs/binfmt_misc/WSLInterop");
  a->container = file_exists("/.dockerenv") || file_exists("/run/.containerenv") ||
                 file_contains("/proc/1/cgroup", "docker") ||
                 file_contains("/proc/1/cgroup", "containerd") ||
                 file_contains("/proc/1/cgroup", "kubepods") ||
                 file_contains("/proc/1/cgroup", "lxc") ||
                 file_contains("/proc/1/cgroup", "libpod");
  a->hypervisor_node = file_exists("/sys/hypervisor/type") ||
                       file_exists("/proc/device-tree/hypervisor/compatible") ||
                       file_exists("/proc/xen/capabilities");
  file_first_line("/sys/class/dmi/id/sys_vendor", a->dmi_vendor, sizeof a->dmi_vendor);
  file_first_line("/sys/class/dmi/id/product_name", a->dmi_product, sizeof a->dmi_product);
  for (i = 0; k_dmi_virtual[i]; ++i) {
    strncpy(lowered, a->dmi_vendor, sizeof lowered - 1); lowered[sizeof lowered - 1] = 0;
    lower_ascii(lowered);
    if (strstr(lowered, k_dmi_virtual[i])) a->dmi_virtual = 1;
    strncpy(lowered, a->dmi_product, sizeof lowered - 1); lowered[sizeof lowered - 1] = 0;
    lower_ascii(lowered);
    if (strstr(lowered, k_dmi_virtual[i])) a->dmi_virtual = 1;
  }
  a->pmu_source = "none";
  for (i = 0; k_pmu_sources[i] && !a->hardware_pmu; ++i) {
    char path[BCIR_MICROBENCH_LINE];
    snprintf(path, sizeof path, "/sys/bus/event_source/devices/%s/type", k_pmu_sources[i]);
    if (file_exists(path)) { a->hardware_pmu = 1; a->pmu_source = k_pmu_sources[i]; }
  }
  a->perf_event_paranoid = file_long("/proc/sys/kernel/perf_event_paranoid", -99);
  file_first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", a->governor,
                  sizeof a->governor);
  file_first_line("/sys/devices/system/clocksource/clocksource0/current_clocksource",
                  a->clocksource, sizeof a->clocksource);
  a->rapl = file_exists("/sys/class/powercap/intel-rapl/enabled") ||
            file_exists("/sys/class/powercap/intel-rapl:0/name");
  a->timer_quantum_ns = observe_timer_quantum();

  a->signals[0] = 0;
  if (a->hypervisor_flag) append_signal(a->signals, sizeof a->signals, "hypervisor-flag");
  if (a->hypervisor_node) append_signal(a->signals, sizeof a->signals, "hypervisor-node");
  if (a->dmi_virtual) append_signal(a->signals, sizeof a->signals, "dmi-virtual");
  if (a->wsl) append_signal(a->signals, sizeof a->signals, "wsl");
  if (a->container) append_signal(a->signals, sizeof a->signals, "container");
  if (a->hypervisor_flag || a->hypervisor_node || a->dmi_virtual || a->wsl)
    a->tenancy = "virtualized";
  else if (a->container)
    a->tenancy = "containerized";
  else if (a->hardware_pmu)
    a->tenancy = "bare-metal";
  else
    a->tenancy = "unproven";
  if (!a->signals[0]) {
    if (a->hardware_pmu) snprintf(a->signals, sizeof a->signals, "pmu=%s", a->pmu_source);
    else strncpy(a->signals, "no PMU exposed", sizeof a->signals - 1);
  }
}

/* --- JSON writers (bounded; strings escaped) --------------------------------------- */

static void json_str(const char *s) {
  putchar('"');
  for (; *s; ++s) {
    unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\') { putchar('\\'); putchar((int)c); }
    else if (c < 0x20 || c == 0x7f) printf("\\u%04x", (unsigned)c);
    else putchar((int)c);
  }
  putchar('"');
}

static void json_samples(const char *key, const uint64_t *xs, int k) {
  int i;
  printf("    \"%s\": [", key);
  for (i = 0; i < k; ++i) printf("%s%" PRIu64, i ? ", " : "", xs[i]);
  printf("],\n");
}

static void json_stat(const char *key, stat4 s) {
  printf("    \"%s\": [%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "],\n", key, s.min,
         s.median, s.max, s.mad);
}

/* q8(ns) = max(256, ns*256/base): a regime is never cheaper than streaming. */
static uint64_t q8(uint64_t ns, uint64_t base) {
  uint64_t v = !base ? Q8 : ns > UINT64_MAX / Q8 ? UINT64_MAX : (ns * Q8) / base;
  if(v<(uint64_t)Q8)v=(uint64_t)Q8;
  return v>(uint64_t)INT64_MAX?(uint64_t)INT64_MAX:v;
}

static int parse_u64(const char *s, uint64_t *out) {
  char *end; unsigned long long value;
  if(!s||!*s||*s=='-')return -1;
  errno=0;value=strtoull(s,&end,10);
  if(errno||*end)return -1;
  *out=(uint64_t)value;return 0;
}

static int parse_i64(const char *s, int64_t *out) {
  char *end; long long value;
  if(!s||!*s)return -1;
  errno=0;value=strtoll(s,&end,10);
  if(errno||*end)return -1;
  *out=(int64_t)value;return 0;
}

static const char *host_os(void) {
#if defined(__linux__)
  return "linux";
#elif defined(__APPLE__)
  return "darwin";
#elif defined(_WIN32)
  return "windows";
#else
  return "unknown";
#endif
}

static const char *host_arch(void) {
#if defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
  return "aarch64";
#elif defined(__i386__) || defined(_M_IX86)
  return "x86";
#elif defined(__arm__)
  return "arm";
#elif defined(__riscv)
  return "riscv";
#else
  return "unknown";
#endif
}

static const char *host_compiler(void) {
#if defined(__VERSION__)
  return __VERSION__;
#else
  return "unknown";
#endif
}

int main(int argc, char **argv) {
  uint64_t parsed_n=1u<<22,parsed_repeats=5;int64_t cal_gen=1;
  if((argc>1&&parse_u64(argv[1],&parsed_n))||(argc>2&&parse_u64(argv[2],&parsed_repeats))||
     (argc>3&&parse_i64(argv[3],&cal_gen))||argc>4||parsed_n>SIZE_MAX||
     parsed_n>BCIR_MICROBENCH_MAX_N||parsed_repeats<1||
     parsed_repeats>BCIR_MICROBENCH_MAX_REPEATS||cal_gen<0){
    fprintf(stderr,"invalid arguments (n<=%" PRIu64 ", repeats<=%" PRIu64 ", cal_gen>=0)\n",
            BCIR_MICROBENCH_MAX_N,BCIR_MICROBENCH_MAX_REPEATS);return 2;}
  size_t n=(size_t)parsed_n;int repeats=(int)parsed_repeats;
  if (n < 2) n = 2;   /* need >=2 elements: the Fisher-Yates loop (n-1 down to 1) underflows and the strided step is reduced mod n */

  if(n>SIZE_MAX/sizeof(double)||n>SIZE_MAX/sizeof(size_t)||
     (size_t)repeats>SIZE_MAX/(5u*sizeof(uint64_t))){fprintf(stderr,"requested benchmark is too large\n");return 2;}

  bcir_host_allocator allocator=bcir_host_allocator_default();
  double *buf = (double *)bcir_host_allocate(&allocator,n * sizeof *buf);
  size_t *perm = (size_t *)bcir_host_allocate(&allocator,n * sizeof *perm);
  /* five sample arrays: four regimes + a sort scratch. */
  uint64_t *samples = (uint64_t *)bcir_host_allocate(&allocator,(size_t)repeats * 5u * sizeof *samples);
  unsigned char *seen = (unsigned char *)bcir_host_allocate(&allocator,(n + 7u) / 8u);
  if (!buf || !perm || !samples || !seen) { fprintf(stderr, "oom\n");
    bcir_host_deallocate(&allocator,buf);bcir_host_deallocate(&allocator,perm);
    bcir_host_deallocate(&allocator,samples);bcir_host_deallocate(&allocator,seen);return 2; }
  uint64_t *s_stream = samples, *s_strided = samples + repeats, *s_random = samples + 2 * repeats,
           *s_compute = samples + 3 * repeats, *scratch = samples + 4 * repeats;

  for (size_t i = 0; i < n; ++i) { buf[i] = (double)(i % 97); perm[i] = i; }
  for (size_t i = n - 1; i > 0; --i) {           /* seeded Fisher-Yates */
    size_t j = (size_t)(lcg() % (i + 1));
    size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }

  /* The census runs before the timed passes: the claim "every element is touched" is
   * counted, not assumed (the old walk touched n/16 and reported nothing). */
  size_t unique_strided = census_strided(seen, n, BCIR_MICROBENCH_STRIDE);
  size_t unique_random = census_random(seen, perm, n);
  size_t unique_stream = n;

  for (int r = 0; r < repeats; ++r) s_stream[r] = pass_stream(buf, n);
  for (int r = 0; r < repeats; ++r) s_strided[r] = pass_strided(buf, n, BCIR_MICROBENCH_STRIDE);
  for (int r = 0; r < repeats; ++r) s_random[r] = pass_random(buf, perm, n);
  for (int r = 0; r < repeats; ++r) s_compute[r] = pass_compute(n);
  stat4 st_stream = summarize(s_stream, repeats, scratch);
  stat4 st_strided = summarize(s_strided, repeats, scratch);
  stat4 st_random = summarize(s_random, repeats, scratch);
  stat4 st_compute = summarize(s_compute, repeats, scratch);

  attestation a;
  attest(&a);
  char provenance[2 * BCIR_MICROBENCH_LINE];
  snprintf(provenance, sizeof provenance,
           "native microbench (%s: %s) n=%zu repeats=%d unique=%zu/%zu/%zu", a.tenancy,
           a.signals, n, repeats, unique_stream, unique_strided, unique_random);

  printf("{\n");
  printf("  \"name\": \"native\",\n");
  printf("  \"cal_gen\": %" PRId64 ",\n", cal_gen);
  printf("  \"samples\": %d,\n", repeats);
  printf("  \"provenance\": "); json_str(provenance); printf(",\n");
  printf("  \"stream_q8\": %d,\n", Q8);
  printf("  \"strided_q8\": %" PRIu64 ",\n", q8(st_strided.median, st_stream.median));
  printf("  \"random_q8\": %" PRIu64 ",\n", q8(st_random.median, st_stream.median));
  printf("  \"compute_q8\": %" PRIu64 ",\n", q8(st_compute.median, st_stream.median));
  printf("  \"evidence\": {\n");
  printf("    \"tenancy\": "); json_str(a.tenancy); printf(",\n");
  printf("    \"signals\": "); json_str(a.signals); printf(",\n");
  printf("    \"hardware_pmu\": %s,\n", a.hardware_pmu ? "true" : "false");
  printf("    \"pmu_source\": "); json_str(a.pmu_source); printf(",\n");
  printf("    \"perf_event_paranoid\": %ld,\n", a.perf_event_paranoid);
  printf("    \"cpufreq_governor\": "); json_str(a.governor); printf(",\n");
  printf("    \"rapl\": %s,\n", a.rapl ? "true" : "false");
  printf("    \"clocksource\": "); json_str(a.clocksource); printf(",\n");
  printf("    \"timer_quantum_ns\": %" PRIu64 ",\n", a.timer_quantum_ns);
  printf("    \"dmi_vendor\": "); json_str(a.dmi_vendor); printf(",\n");
  printf("    \"dmi_product\": "); json_str(a.dmi_product); printf(",\n");
  printf("    \"os\": "); json_str(host_os()); printf(",\n");
  printf("    \"arch\": "); json_str(host_arch()); printf(",\n");
  printf("    \"compiler\": "); json_str(host_compiler()); printf(",\n");
  printf("    \"n\": %zu,\n", n);
  printf("    \"repeats\": %d,\n", repeats);
  printf("    \"stride\": %zu,\n", BCIR_MICROBENCH_STRIDE);
  printf("    \"unique_stream\": %zu,\n", unique_stream);
  printf("    \"unique_strided\": %zu,\n", unique_strided);
  printf("    \"unique_random\": %zu,\n", unique_random);
  printf("    \"working_set_bytes\": %zu,\n", unique_strided * sizeof(double));
  printf("    \"outlier_policy\": \"median of repeats; no sample discarded; dispersion as the MAD\",\n");
  json_samples("stream_ns", s_stream, repeats);
  json_samples("strided_ns", s_strided, repeats);
  json_samples("random_ns", s_random, repeats);
  json_samples("compute_ns", s_compute, repeats);
  json_stat("stream_stat", st_stream);
  json_stat("strided_stat", st_strided);
  json_stat("random_stat", st_random);
  printf("    \"compute_stat\": [%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "]\n",
         st_compute.min, st_compute.median, st_compute.max, st_compute.mad);
  printf("  }\n");
  printf("}\n");

  bcir_host_deallocate(&allocator,buf);bcir_host_deallocate(&allocator,perm);
  bcir_host_deallocate(&allocator,samples);bcir_host_deallocate(&allocator,seen);
  return 0;
}
