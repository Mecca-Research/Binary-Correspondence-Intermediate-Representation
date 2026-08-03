#!/usr/bin/env bash
# probe_capabilities.sh -- what a candidate host can actually support, as a checkable record.
#
# BCIR's remaining phases are blocked on ACCESS rather than on code. J6's hardware counters
# need a PMU, J7's driver experiment needs to load a kernel module and bind a device, and
# every timing claim wants a core whose frequency does not move underneath it. Which of those
# a machine offers is not obvious from what the machine is called: this container runs as root
# with cap_sys_module, cap_sys_rawio and cap_perfmon and still cannot count a single cycle,
# because the hypervisor exposes no PMU at all. Privilege and capability are different things,
# and only one of them can be granted.
#
# So this probes rather than assumes, and prints a JSON object. Run it on any candidate --
# this container, the phone under Termux, a bare-metal box being evaluated -- and the records
# are comparable. docs/BCIR_TARGET_ACCESS.md says which capability each open phase needs.
#
# It reads. It does not configure anything, load anything, or write to /dev/mem: a probe that
# changed the machine would be measuring a machine that no longer exists.
#
# Portable to Termux on purpose, which is why there is no awk, seq, sed or taskset here.
set -uo pipefail          # NOT -e: a probe's job is to report a failure, not to die of one.

emit() { printf '  "%s": %s,\n' "$1" "$2"; }
emit_s() { printf '  "%s": "%s",\n' "$1" "$2"; }
yn() { if [ "$1" = "yes" ]; then printf 'true'; else printf 'false'; fi; }

have_file() { [ -e "$1" ] && echo yes || echo no; }
read_or() { if [ -r "$1" ]; then cat "$1" 2>/dev/null | head -1; else echo "$2"; fi; }

printf '{\n'

# --- identity -----------------------------------------------------------------------------
emit_s "arch" "$(uname -m)"
emit_s "kernel" "$(uname -r)"
emit_s "os" "$(uname -s)"
virt="unknown"
if command -v systemd-detect-virt >/dev/null 2>&1; then virt="$(systemd-detect-virt 2>/dev/null || echo none)"; fi
if [ "${virt}" = "unknown" ] && [ -r /proc/cpuinfo ] && grep -q hypervisor /proc/cpuinfo 2>/dev/null; then
  virt="hypervisor-flag"
fi
emit_s "virtualization" "${virt}"
emit_s "uid" "$(id -u)"

# --- counters: the J6 blocker -------------------------------------------------------------
#
# TWO independent things must hold, and they fail differently. `perf_event_paranoid` is a
# POLICY -- Android ships 3, which denies user-space counting outright. An event source named
# `cpu` under /sys/bus/event_source/devices is the HARDWARE -- without it perf_event_open
# returns ENOENT no matter how permissive the policy is, which is this container's situation.
# Reporting them separately is the point: one is granted, the other is provisioned.
emit_s "perf_event_paranoid" "$(read_or /proc/sys/kernel/perf_event_paranoid na)"
sources="none"
if [ -d /sys/bus/event_source/devices ]; then
  sources="$(ls /sys/bus/event_source/devices 2>/dev/null | tr '\n' ' ')"
fi
emit_s "perf_event_sources" "${sources}"
pmu=no
for candidate in /sys/bus/event_source/devices/cpu \
                 /sys/bus/event_source/devices/cpu_core \
                 /sys/bus/event_source/devices/armv8_pmuv3 \
                 /sys/bus/event_source/devices/armv8_pmuv3_0; do
  [ -d "${candidate}" ] && pmu=yes
done
emit "hardware_pmu" "$(yn ${pmu})"

# --- frequency: what makes a nanosecond mean the same thing twice --------------------------
emit "cpufreq_exposed" "$(yn "$(have_file /sys/devices/system/cpu/cpu0/cpufreq)")"
emit_s "cpufreq_governor" "$(read_or /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor na)"
emit "turbo_control" "$(yn "$(have_file /sys/devices/system/cpu/intel_pstate/no_turbo)")"
emit_s "clocksource" "$(read_or /sys/devices/system/clocksource/clocksource0/current_clocksource na)"

# --- isolation: what makes a measured core quiet -------------------------------------------
emit_s "cpus" "$(read_or /sys/devices/system/cpu/online na)"
emit_s "isolated_cpus" "$(read_or /sys/devices/system/cpu/isolated '')"
emit_s "nohz_full" "$(read_or /sys/devices/system/cpu/nohz_full '')"
pin=no
python3 -c 'import os; os.sched_setaffinity(0, {list(os.sched_getaffinity(0))[0]})' 2>/dev/null && pin=yes
emit "can_pin_cpu" "$(yn ${pin})"
fifo=no
chrt -f 1 true >/dev/null 2>&1 && fifo=yes
emit "can_sched_fifo" "$(yn ${fifo})"

# --- driver work: the J7 blockers ----------------------------------------------------------
#
# A resident driver has to be BUILT against the running kernel and LOADED, then bind something.
# cap_sys_module without /lib/modules and headers is a permission to do a thing the machine
# has not been given the parts for -- which is exactly this container.
mods="no"
[ -d /lib/modules/"$(uname -r)" ] && mods="yes"
emit "kernel_modules_tree" "$(yn ${mods})"
hdrs="no"
{ [ -d /usr/src/linux-headers-"$(uname -r)" ] || [ -d /lib/modules/"$(uname -r)"/build ]; } && hdrs="yes"
emit "kernel_headers" "$(yn ${hdrs})"
emit "dev_mem" "$(yn "$(have_file /dev/mem)")"
emit "dev_msr" "$(yn "$(have_file /dev/cpu/0/msr)")"
emit "vfio" "$(yn "$(have_file /dev/vfio/vfio)")"
emit "uio" "$(yn "$(have_file /dev/uio0)")"
iommu=no
{ [ -d /sys/class/iommu ] && [ -n "$(ls /sys/class/iommu 2>/dev/null)" ]; } && iommu=yes
emit "iommu" "$(yn ${iommu})"

# --- memory ---------------------------------------------------------------------------------
emit_s "hugepages" "$(read_or /proc/sys/vm/nr_hugepages na)"
lock=no
python3 -c 'import ctypes,sys; sys.exit(0 if ctypes.CDLL("libc.so.6").mlockall(3)==0 else 1)' 2>/dev/null && lock=yes
emit "can_mlockall" "$(yn ${lock})"
nodes=0
[ -d /sys/devices/system/node ] && nodes="$(ls -d /sys/devices/system/node/node* 2>/dev/null | wc -l | tr -d ' ')"
emit "numa_nodes" "${nodes:-0}"

# --- contention accounting, which the calibration reader needs ------------------------------
emit "proc_stat_readable" "$(yn "$( [ -r /proc/stat ] && echo yes || echo no )")"
cg=no
{ [ -r /sys/fs/cgroup/cpu.stat ] || [ -r /sys/fs/cgroup/cpu/cpu.stat ]; } && cg=yes
emit "cgroup_cpu_stat_readable" "$(yn ${cg})"

# --- the clock the harness will actually get ------------------------------------------------
tick="$(python3 - 2>/dev/null <<'PY'
import time
deltas = []
for _ in range(20000):
    a = time.perf_counter_ns(); b = time.perf_counter_ns()
    if b > a: deltas.append(b - a)
print(min(deltas) if deltas else -1)
PY
)"
emit "observed_clock_tick_ns" "${tick:--1}"

# Last entry: no trailing comma.
printf '  "probe_version": 1\n}\n'
