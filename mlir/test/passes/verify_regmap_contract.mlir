// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// The UART blueprint's U0: the RegMap contract law -- the R12 lowering-contract discipline applied
// to a device's register map. A bcir.regmap_contract must name a placement of the U0 registry and
// lay the normative register table (RBR THR IER IIR FCR LCR MCR LSR MSR SCR DLL DLM) out as that
// placement does: its stride, base offset, FCR reset and reference clock, and each register at
// base_offset + slot * stride. The registry is generated from bcir/frontends/devices/uart_model.py
// (BCIRUartPlacements.inc). Mirrors the oracle's verify_regmap_contract and uart_laws' literal
// attestation; vacuous when no contract op is present.

// (1 -- NEGATIVE) a placement the registry does not hold.
bcir.module @r3uart_unknown_placement {
  // expected-error @+1 {{R3-uart: regmap_contract u names an unknown placement 'pc16550x' (registry: axi16450, axi16550, core16550_apb, pc16550d)}}
  bcir.regmap_contract @u { placement = "pc16550x", stride = 1 : i32, base_offset = 0 : i64,
                            fcr_reset = 0 : i32, ref_clk_hz = 1843200 : i64,
                            reg_offsets = array<i64: 0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 0, 1> }
}

// -----

// (2 -- NEGATIVE) hazard H-A: a byte-strided layout claimed for the 4-byte-strided Core16550.
bcir.module @r3uart_stride_lie {
  // expected-error @+2 {{R3-uart: regmap_contract u: stride: contract 0x1 != placement 0x4}}
  // expected-error @+1 {{R3-uart: regmap_contract u: register IER: contract 0x1 != laid out 0x4}}
  bcir.regmap_contract @u { placement = "core16550_apb", stride = 1 : i32, base_offset = 0 : i64,
                            fcr_reset = 1 : i32, ref_clk_hz = 18432000 : i64,
                            reg_offsets = array<i64: 0, 0, 1, 8, 8, 12, 16, 20, 24, 28, 0, 4> }
}

// -----

// (3 -- NEGATIVE) the AXI core's registers start at 0x1000, not at the device base.
bcir.module @r3uart_base_lie {
  // expected-error @+1 {{R3-uart: regmap_contract u: base offset: contract 0x0 != placement 0x1000}}
  bcir.regmap_contract @u { placement = "axi16550", stride = 4 : i32, base_offset = 0 : i64,
                            fcr_reset = 0 : i32, ref_clk_hz = 100000000 : i64,
                            reg_offsets = array<i64: 4096, 4096, 4100, 4104, 4104, 4108, 4112, 4116, 4120, 4124, 4096, 4100> }
}

// -----

// (4 -- NEGATIVE) Core16550's FCR resets to 01 (FIFOs on at reset), and its reference is 18.432 MHz.
bcir.module @r3uart_reset_and_clock_lies {
  // expected-error @+2 {{R3-uart: regmap_contract u: fcr_reset 0x00 != placement 0x01}}
  // expected-error @+1 {{R3-uart: regmap_contract u: ref_clk_hz 1843200 != placement 18432000}}
  bcir.regmap_contract @u { placement = "core16550_apb", stride = 4 : i32, base_offset = 0 : i64,
                            fcr_reset = 0 : i32, ref_clk_hz = 1843200 : i64,
                            reg_offsets = array<i64: 0, 0, 4, 8, 8, 12, 16, 20, 24, 28, 0, 4> }
}

// -----

// (5 -- NEGATIVE) one register out of place: LSR at slot 6, where MSR is.
bcir.module @r3uart_register_lie {
  // expected-error @+1 {{R3-uart: regmap_contract u: register LSR: contract 0x6 != laid out 0x5}}
  bcir.regmap_contract @u { placement = "pc16550d", stride = 1 : i32, base_offset = 0 : i64,
                            fcr_reset = 0 : i32, ref_clk_hz = 1843200 : i64,
                            reg_offsets = array<i64: 0, 0, 1, 2, 2, 3, 4, 6, 6, 7, 0, 1> }
}

// -----

// (6 -- NEGATIVE) a contract that lays out fewer registers than the normative table has.
bcir.module @r3uart_register_count {
  // expected-error @+1 {{R3-uart: regmap_contract u: registers: contract lays out 11; the model has 12}}
  bcir.regmap_contract @u { placement = "pc16550d", stride = 1 : i32, base_offset = 0 : i64,
                            fcr_reset = 0 : i32, ref_clk_hz = 1843200 : i64,
                            reg_offsets = array<i64: 0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 0> }
}

// -----

// (7 -- POSITIVE) the truthful records: the byte-strided PC16550D and the AXI core's 16450 build.
bcir.module @r3uart_legal_contracts {
  bcir.regmap_contract @pc { placement = "pc16550d", stride = 1 : i32, base_offset = 0 : i64,
                             fcr_reset = 0 : i32, ref_clk_hz = 1843200 : i64,
                             reg_offsets = array<i64: 0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 0, 1> }
  bcir.regmap_contract @axi { placement = "axi16450", stride = 4 : i32, base_offset = 4096 : i64,
                              fcr_reset = 0 : i32, ref_clk_hz = 100000000 : i64,
                              reg_offsets = array<i64: 4096, 4096, 4100, 4104, 4104, 4108, 4112, 4116, 4120, 4124, 4096, 4100> }
}

// -----

// (8 -- NEGATIVE, structural) a stride no register lane can have is refused at parse time.
bcir.module @r3uart_bad_stride {
  // expected-error @+1 {{'bcir.regmap_contract' op stride 3 is not 1, 2, 4 or 8}}
  bcir.regmap_contract @u { placement = "pc16550d", stride = 3 : i32, base_offset = 0 : i64,
                            fcr_reset = 0 : i32, ref_clk_hz = 1843200 : i64,
                            reg_offsets = array<i64: 0> }
}

// -----

// (9 -- NEGATIVE, structural) a base offset below the device base is refused, never wrapped.
bcir.module @r3uart_negative_base {
  // expected-error @+1 {{'bcir.regmap_contract' op base_offset -4096 is not a non-negative multiple of the stride}}
  bcir.regmap_contract @u { placement = "axi16550", stride = 4 : i32, base_offset = -4096 : i64,
                            fcr_reset = 0 : i32, ref_clk_hz = 100000000 : i64,
                            reg_offsets = array<i64: 0> }
}

// -----

// (10 -- NEGATIVE, structural) a base offset between two register lanes.
bcir.module @r3uart_unaligned_base {
  // expected-error @+1 {{'bcir.regmap_contract' op base_offset 4098 is not a non-negative multiple of the stride}}
  bcir.regmap_contract @u { placement = "axi16550", stride = 4 : i32, base_offset = 4098 : i64,
                            fcr_reset = 0 : i32, ref_clk_hz = 100000000 : i64,
                            reg_offsets = array<i64: 0> }
}

// -----

// (11 -- NEGATIVE, structural) FCR is a byte register: its reset value is a byte.
bcir.module @r3uart_wide_fcr_reset {
  // expected-error @+1 {{'bcir.regmap_contract' op fcr_reset 256 is not a byte}}
  bcir.regmap_contract @u { placement = "pc16550d", stride = 1 : i32, base_offset = 0 : i64,
                            fcr_reset = 256 : i32, ref_clk_hz = 1843200 : i64,
                            reg_offsets = array<i64: 0> }
}

// -----

// (12 -- NEGATIVE, structural) a baud reference clock of 0 Hz divides nothing.
bcir.module @r3uart_zero_clock {
  // expected-error @+1 {{'bcir.regmap_contract' op ref_clk_hz 0 is not positive}}
  bcir.regmap_contract @u { placement = "pc16550d", stride = 1 : i32, base_offset = 0 : i64,
                            fcr_reset = 0 : i32, ref_clk_hz = 0 : i64,
                            reg_offsets = array<i64: 0> }
}

// -----

// (13 -- NEGATIVE, structural) a contract that lays out no register attests nothing.
bcir.module @r3uart_no_registers {
  // expected-error @+1 {{'bcir.regmap_contract' op reg_offsets is empty: the contract lays out no register}}
  bcir.regmap_contract @u { placement = "pc16550d", stride = 1 : i32, base_offset = 0 : i64,
                            fcr_reset = 0 : i32, ref_clk_hz = 1843200 : i64,
                            reg_offsets = array<i64> }
}

// -----

// (14 -- NEGATIVE, structural) a register below the device base.
bcir.module @r3uart_negative_register {
  // expected-error @+1 {{'bcir.regmap_contract' op register 2 has a negative offset}}
  bcir.regmap_contract @u { placement = "pc16550d", stride = 1 : i32, base_offset = 0 : i64,
                            fcr_reset = 0 : i32, ref_clk_hz = 1843200 : i64,
                            reg_offsets = array<i64: 0, 0, -1, 2, 2, 3, 4, 5, 6, 7, 0, 1> }
}
