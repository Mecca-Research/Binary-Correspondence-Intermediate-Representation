# BCIR root of trust, TPM and the measured kernel — the security workload

> **Status: research plan (2026-11-07); nothing built.** The owner's direction: define how a silicon
> root of trust (RoT) with a Trusted Platform Module (TPM) carries BCIR's security workload, from
> pre-boot to the running kernel. The mechanisms below are candidates, each cited to its public
> source; RoT0's research slice decides between them before anything is built. Roadmaps own the
> future, [`DEVELOPMENT_HISTORY.md`](../DEVELOPMENT_HISTORY.md) the past. This program sits beside
> the firmware program ([driver roadmap §5.7](BCIR_DRIVER_KERNEL_ROADMAP.md#57-firmware-program-uefi-acpi-smbios-and-tpm-planned))
> and follows its order (§5.6); it is not started.

## 1. Who does what

| Role | Who plays it | What it does |
|---|---|---|
| **Root of trust for measurement** | the silicon RoT: an integrated block such as OpenTitan or Caliptra on an SoC, or the platform's own on a PC | verifies and measures the first mutable code before the CPUs run it; on an SoC, derives layered identities (TCG DICE) |
| **Roots of trust for storage and reporting** | a TPM 2.0: discrete on SPI, LPC or I2C, a firmware TPM, or integrated | holds the PCRs and the event log's anchor, quotes them, seals keys to a policy over them, and locks out repeated authorization failures |
| **The static chain** | firmware (§5.7): SEC, PEI, DXE, BDS, the boot manager | measures each stage before it runs and enforces the Secure Boot policy |
| **BCIR** | the loader, the runtime, its toolchain | computes every expected measurement offline from content-addressed artifacts, holds the measurement map as a generated registry, checks the trust laws before an artifact activates, drives the TPM as a driver package, and measures every artifact it activates |

BCIR does not reimplement a TPM or a RoT. It is the constructor and the verifier of what gets
measured, the same discipline as its other rails: one authoritative model, generated views, laws
that refuse before cost.

## 2. The chain, stage by stage

PCR numbers follow the TCG PC Client Platform Firmware Profile's PCR usage table for firmware
(PCRs 0–7) and dynamic launch (17–22), and the operating system's conventions for PCRs 8–15.

| Stage | Verified by | Measured into | BCIR's part |
|---|---|---|---|
| **Pre-boot** — the RoT releases the CPUs | the RoT's ROM verifies the first mutable firmware | PCR 0 (the static root's code); on SoCs, the DICE layers | records the RoT's measurements in the manifest |
| **Firmware** | Secure Boot and the platform's signing | PCRs 0–6: firmware code and configuration, drivers and option ROMs, the boot manager, the partition table | firmware program stages A–B |
| **Secure Boot policy** | PK, KEK, db, dbx (and SBAT revocation where shim boots) | PCR 7 | the policy is a record of the manifest |
| **The kernel image** | Secure Boot verifies **one** signed PE: a unified kernel image | the stub measures each section into PCR 11; a command line passed from outside into PCR 12 | BCIR computes the expected PCR 11 at build time and ships it signed |
| **Kernel and runtime integrity** | appraisal: a signature or hash per executable, module and artifact | an integrity log extended into an OS PCR (Linux's IMA uses PCR 10) | each artifact's activation is one measurement event (§3.3) |
| **Dynamic launch (DRTM)** | the CPU's launch instruction: Intel TXT's `GETSEC[SENTER]`, AMD's `SKINIT`, Arm's DRTM | resets PCRs 17–22 from a dynamic locality, then measures what it launches into them | the BCIR runtime as the launched environment (RoT8) |
| **Attestation** | a verifier outside the machine | a TPM quote over the PCRs, with the event log that replays to them | the manifest is the verifier's reference values |

**Unified kernel images.** A unified kernel image (UKI) joins the kernel, its initrd, its command
line and the boot stub into one signed UEFI PE, so the RoT's chain verifies and measures one
contiguous artifact. The UAPI group's specification has the stub measure the sections in a
canonical order into PCR 11. It carries a signed set of expected PCR 11 values (`.pcrsig`) and the
key that signs them (`.pcrpkey`), so a disk key can be sealed to the signer rather than to one
image's hash. systemd-stub is the reference stub, and `systemd-measure` computes the expected
values. For BCIR-Linux the UKI is the image. For the native runtime, RoT3 decides between a UKI
whose sections carry the BCIR loader, runtime and initial execution portfolio, which standard
tooling can then verify and measure, and a BCAB bundle inside one such section.

**Dynamic launch.** A DRTM re-establishes trust after firmware ran, so firmware drops out of the
launched kernel's trusted base. PCRs 17–22 hold all ones at boot and only the launch resets them,
so a quote over them proves the launch happened. On Linux, TrenchBoot's Secure Launch series
(Intel TXT first) was at v16 in May 2026 and not yet merged. QEMU offers no dynamic launch, so this
stage needs physical rigs.

## 3. What BCIR defines

### 3.1 The trust registry
One model, generated views, as `uart_model.py` is for the UART:
- **The PCR map:** each measured artifact class with its PCR, event type and hash banks, for the
  firmware range, the OS range and the dynamic range. Machine-readable, as the firmware program
  requires.
- **The TPM's placements:** the PC Client interfaces of the TCG PC Client Platform TPM Profile —
  TIS (FIFO) and CRB at 0xFED4_0000, one 4 KiB page per locality — and the I2C and SPI
  transports.

The registry generates the C and C++ tables the runtime, the twin and the law rail read. Its
`--check` holds them current.

### 3.2 The measurement manifest
Every artifact the platform runs has an entry: its content address, its PCR and its event. That
covers the UKI's sections, the BCIR loader, driver packages, StreamPacks, plans, frozen priors and
model weights. BCIR replays the TPM's extend over the manifest offline and predicts each PCR, on
the oracle and the C twin, held in parity like every other rail. The predicted values are what
`.pcrsig` signs and what a verifier checks quotes against.

### 3.3 The trust laws (candidates)
Advisory first, as R3-uart is, then gating once the rungs below give evidence:
1. **Measure before execute.** No artifact activates before its measurement is extended and logged.
2. **The log replays.** The event log replays, bank by bank, to the PCR values a quote reports.
3. **Seal to a policy, not a hash.** A secret is sealed under a signed policy over the PCRs that
   cover its consumer (the TPM's `PolicyAuthorize`), so an update re-signs instead of re-sealing.
   A modified binary, boot flag or command line releases nothing.
4. **Dynamic PCRs come only from a dynamic launch.** PCRs 17–22 count only after the launch event.
5. **No secret crosses the TPM bus in the clear.** A command carrying a secret runs in a salted
   HMAC session with parameter encryption. Linux 6.10's TPM bus protection (`CONFIG_TCG_TPM2_HMAC`)
   is the reference. It defends against an interposer on the SPI or LPC bus, which an integrated
   or firmware TPM removes.
6. **Activation is a measurement.** BCIR promotes artifacts only at quiescent generation
   boundaries (invariant 6), and each promotion or rollback is one logged measurement event. None
   is silent.

### 3.4 The TPM driver package
The TPM is a device like the UART and climbs the same D0–D7 ladder:
- the TIS and CRB interfaces, localities, command and response framing, and timeouts on the
  event-timer substrate;
- polled first, so the early-boot driver runs before interrupts exist;
- simulated against swtpm (libtpms) and the TCG's reference implementation, as `sim16550` stands
  in for the UART;
- law 5's sessions in its command path.

Its MMIO interface needs no DMA, so it can follow the event timers early and give the firmware
program a tested driver before stage A (§5.6).

### 3.5 Three lockouts, kept apart
"Lockout" names three mechanisms that a design must not conflate:
- **Unseal refusal:** the policy does not match, so a modified binary, boot flag or hash releases
  nothing. This one is automatic.
- **The TPM's dictionary-attack lockout:** repeated authorization failures lock the TPM out.
- **Appraisal refusal:** a modified artifact does not run.

Each has a recovery path that an authorized update or re-enrollment takes: a recovery key, and
A/B images. A legitimate update must never brick a machine.

## 4. Emulation first: debugging without bricking

| Level | What runs | What it can debug |
|---|---|---|
| **E0 — host models** | QEMU's TPM TIS/CRB devices backed by swtpm, OVMF with TPM 2.0 and Secure Boot, the tpm2 tools | the whole measured boot, sealing, lockout and quotes on the cloud host |
| **E1 — RTL simulation** | OpenTitan's Earl Grey in Verilator (its ROM boots an image from simulated flash); Caliptra's RTL and firmware emulator; a VexRiscv SoC in LiteX's Verilator simulation running Linux | a RoT's boot and DICE layering, and an early-boot TPM driver at register level |
| **E2 — FPGA** | LiteX boards; OpenTitan's FPGA targets; LitePCIe for a PCIe-attached RoT | timing, buses and resets at speed |
| **E3 — physical** | rigs with a recovery path: an external flash programmer or dual images, Secure Boot in setup mode with test keys first, and vendors' public developer kits where TXT, SKINIT or a vendor RoT needs them | DRTM, real buses and real lockout behavior |

Full-SoC Verilator builds are heavy, so CI carries them, not the bounded local host (AGENTS.md).

## 5. The rungs

| Rung | Deliverable | Exit evidence |
|---|---|---|
| **RoT0 — research contract** | the threat model (bus interposer, evil maid, rollback, quote replay, the gap between measuring and executing); a RoT and TPM choice per target class; the sources | reviewed decisions on §6's questions |
| **RoT1 — registry and manifest** | §3.1–§3.2 with offline PCR prediction on both rails | prediction matches swtpm's PCRs for a scripted event log |
| **RoT2 — TPM driver** | §3.4 against swtpm under QEMU (E0) | D2: polled TIS/CRB parity, localities, timeouts, session-protected commands |
| **RoT3 — measured boot** | the BCIR loader and a UKI on QEMU + OVMF + swtpm | the event log replays to the PCRs; an unsigned image is refused |
| **RoT4 — sealing and lockout** | §3.3 law 3 and §3.5 | a changed kernel, command line or flag unseals nothing; a re-signed update does; recovery works |
| **RoT5 — runtime integrity** | an activation log and appraisal over BCIR artifacts; the Linux rail compared through IMA, IPE and dm-verity | a modified artifact does not activate; the log replays |
| **RoT6 — attestation** | quote, event log and a verifier with the IETF RATS roles (RFC 9334) | a verifier accepts the reference state and refuses any other |
| **RoT7 — RoT simulation** | OpenTitan and Caliptra in Verilator (E1); DICE layers mapped onto the manifest | the RoT's measurements land in the manifest, reproducibly |
| **RoT8 — dynamic launch** | TXT, SKINIT or Arm DRTM on a physical rig (E3), tracking TrenchBoot upstream | a launch resets PCRs 17–22 and the launched runtime's measurement is quoted |

## 6. Open questions RoT0 answers
- **Which root of trust per target class:** a discrete TPM, a firmware TPM or an integrated RoT;
  OpenTitan or Caliptra. Decided by threat model and platform: PC, datacenter SoC or embedded.
- **Where BCIR's artifact measurements go** in the OS range without colliding with the
  conventions Linux and systemd already use. The UAPI group keeps a registry of Linux PCR usage.
- **UKI sections or a BCAB bundle** for the native runtime's image (§2).
- **Whether DRTM earns its cost** on each target, given how large its firmware trusted base is.
- **Post-quantum signatures:** Caliptra signs firmware with ML-DSA; what the TPM side offers is to
  be researched, not assumed.

## 7. Sources
Cited by section when a rung uses them:
- the TCG PC Client Platform Firmware Profile, PC Client Platform TPM Profile, TPM 2.0 Library and
  D-RTM Architecture specifications;
- UEFI 2.11;
- the UAPI group's Unified Kernel Image specification and `systemd-stub(7)`;
- the Linux kernel's TPM security documentation (bus protection, Linux 6.10) and its IMA
  documentation;
- TrenchBoot's Secure Launch patch series;
- the Intel TXT Software Development Guide, the AMD64 Architecture Programmer's Manual Volume 2
  (`SKINIT`) and Arm's DRTM architecture specification;
- the OpenTitan and Caliptra documentation and repositories;
- LiteX and linux-on-litex-vexriscv;
- swtpm and libtpms, and QEMU's TPM documentation;
- IETF RFC 9334.

Material distributed under confidentiality terms stays out, per the driver roadmap's §8.3.
