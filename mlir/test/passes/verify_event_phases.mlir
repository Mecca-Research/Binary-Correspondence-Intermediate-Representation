// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// Part VII A1+B1: EVENT PHASES on the law rail -- first-class asynchronous entry.
// EV1: an event phase declares no phase deps (its trigger is the event source, never
// program order). EV2: the source must be ARMED by an explicit irq.unmask claim in the
// program flow. EV3 (the interrupt-context ordering seam): a resource the handler
// writes may be touched by program claims only inside a masked window or atomically.
// Mirrors kbcir/events.py check_event_phases; vacuous when no phase carries an event.

// (1 -- POSITIVE) the U4 fixture: 16550 RX handler fills the ring; the program-side
// consumer drains it inside an explicit irq.mask/irq.unmask window. No diagnostics.
bcir.module @uart_rx_lawful {
  bcir.registry @RES {
    bcir.resource @IER  { rid = 1 : i32, domain_kind = #bcir.domain<mmio>, shape = array<i64: 1>, layout = #bcir.layout<soa> }
    bcir.resource @IIR  { rid = 2 : i32, domain_kind = #bcir.domain<mmio>, shape = array<i64: 1>, layout = #bcir.layout<soa> }
    bcir.resource @RBR  { rid = 3 : i32, domain_kind = #bcir.domain<mmio>, shape = array<i64: 1>, layout = #bcir.layout<soa> }
    bcir.resource @RING { rid = 4 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 16>, layout = #bcir.layout<soa> }
    bcir.resource @HEAD { rid = 5 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1>, layout = #bcir.layout<soa> }
    bcir.resource @TAIL { rid = 6 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1>, layout = #bcir.layout<soa> }
  }
  bcir.phase @init { id = 0 : i32, deps = [] }
  bcir.phase @drain { id = 1 : i32, deps = [@init] }
  bcir.phase @isr { id = 10 : i32, deps = [], event = "uart_rx" }
  bcir.claim @arm attributes {
    claim_id = 1 : i32, phase = @init, op = "irq.unmask:uart_rx", reads = [],
    writes = [@IER], count = 1 : i64, lane = #bcir.lane<u>,
    stride_class = #bcir.stride_class<scalar>, stride_k = 1 : i32,
    domain = #bcir.domain<mmio>, hazard = #bcir.hazard<atomic>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, is_volatile = true
  } { %i = bcir.index_range 0 to 1 step 1 }
  bcir.claim @mask attributes {
    claim_id = 2 : i32, phase = @drain, op = "irq.mask:uart_rx", reads = [],
    writes = [@IER], count = 1 : i64, lane = #bcir.lane<u>,
    stride_class = #bcir.stride_class<scalar>, stride_k = 1 : i32,
    domain = #bcir.domain<mmio>, hazard = #bcir.hazard<atomic>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, is_volatile = true
  } { %i = bcir.index_range 0 to 1 step 1 }
  bcir.claim @consume attributes {
    claim_id = 3 : i32, phase = @drain, op = "uart.consume", reads = [@RING, @HEAD],
    writes = [@TAIL], count = 1 : i64, lane = #bcir.lane<u>,
    stride_class = #bcir.stride_class<scalar>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1 step 1 }
  bcir.claim @unmask attributes {
    claim_id = 4 : i32, phase = @drain, op = "irq.unmask:uart_rx", reads = [],
    writes = [@IER], count = 1 : i64, lane = #bcir.lane<u>,
    stride_class = #bcir.stride_class<scalar>, stride_k = 1 : i32,
    domain = #bcir.domain<mmio>, hazard = #bcir.hazard<atomic>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, is_volatile = true
  } { %i = bcir.index_range 0 to 1 step 1 }
  // The handler splits its MMIO read from its RAM store -- on the law rail a claim
  // never mixes a device-isolated domain with host memory (the R3 redirection gap).
  bcir.claim @rx_read attributes {
    claim_id = 100 : i32, phase = @isr, op = "uart.rbr_read", reads = [@RBR],
    writes = [], count = 1 : i64, lane = #bcir.lane<u>,
    stride_class = #bcir.stride_class<scalar>, stride_k = 1 : i32,
    domain = #bcir.domain<mmio>, hazard = #bcir.hazard<barriered>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, is_volatile = true
  } { %i = bcir.index_range 0 to 1 step 1 }
  bcir.claim @rx_store attributes {
    claim_id = 101 : i32, phase = @isr, op = "uart.rx_fill", reads = [],
    writes = [@RING, @HEAD], count = 1 : i64, lane = #bcir.lane<u>,
    stride_class = #bcir.stride_class<scalar>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<barriered>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1 step 1 }
}

// -----

// (2 -- NEGATIVE, EV1) deps on an event phase: "interrupts are subroutines" is a lie.
bcir.module @deps_on_event {
  bcir.registry @RES {
    bcir.resource @IER { rid = 1 : i32, domain_kind = #bcir.domain<mmio>, shape = array<i64: 1>, layout = #bcir.layout<soa> }
  }
  bcir.phase @init { id = 0 : i32, deps = [] }
  // expected-error @+1 {{R3: EV1: event phase @isr (source 'uart_rx') declares phase deps -- asynchronous entry has no program-order predecessor (hazards + masking order it)}}
  bcir.phase @isr { id = 10 : i32, deps = [@init], event = "uart_rx" }
  bcir.claim @arm attributes {
    claim_id = 1 : i32, phase = @init, op = "irq.unmask:uart_rx", reads = [],
    writes = [@IER], count = 1 : i64, lane = #bcir.lane<u>,
    stride_class = #bcir.stride_class<scalar>, stride_k = 1 : i32,
    domain = #bcir.domain<mmio>, hazard = #bcir.hazard<atomic>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, is_volatile = true
  } { %i = bcir.index_range 0 to 1 step 1 }
}

// -----

// (3 -- NEGATIVE, EV2) an unarmed source: no irq.unmask claim anywhere in the program
// flow -- enablement must be explicit, never implicit.
bcir.module @unarmed_source {
  bcir.registry @RES {
    bcir.resource @RING { rid = 4 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 16>, layout = #bcir.layout<soa> }
  }
  bcir.phase @init { id = 0 : i32, deps = [] }
  // expected-error @+1 {{R3: EV2: event source 'uart_rx' is never armed -- enablement must be an explicit irq.unmask:uart_rx claim in the program flow, never implicit}}
  bcir.phase @isr { id = 10 : i32, deps = [], event = "uart_rx" }
  bcir.claim @fill attributes {
    claim_id = 101 : i32, phase = @isr, op = "uart.rx_fill", reads = [],
    writes = [@RING], count = 1 : i64, lane = #bcir.lane<u>,
    stride_class = #bcir.stride_class<scalar>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<barriered>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1 step 1 }
}

// -----

// (4 -- NEGATIVE, EV3/B1) the interrupted flow touches the handler-written ring with
// no mask window and no atomic -- the lost-update / torn-read seam, refused.
bcir.module @unmasked_touch {
  bcir.registry @RES {
    bcir.resource @IER  { rid = 1 : i32, domain_kind = #bcir.domain<mmio>, shape = array<i64: 1>, layout = #bcir.layout<soa> }
    bcir.resource @RING { rid = 4 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 16>, layout = #bcir.layout<soa> }
    bcir.resource @TAIL { rid = 6 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1>, layout = #bcir.layout<soa> }
  }
  bcir.phase @init { id = 0 : i32, deps = [] }
  bcir.phase @drain { id = 1 : i32, deps = [@init] }
  bcir.phase @isr { id = 10 : i32, deps = [], event = "uart_rx" }
  bcir.claim @arm attributes {
    claim_id = 1 : i32, phase = @init, op = "irq.unmask:uart_rx", reads = [],
    writes = [@IER], count = 1 : i64, lane = #bcir.lane<u>,
    stride_class = #bcir.stride_class<scalar>, stride_k = 1 : i32,
    domain = #bcir.domain<mmio>, hazard = #bcir.hazard<atomic>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, is_volatile = true
  } { %i = bcir.index_range 0 to 1 step 1 }
  // expected-error @+1 {{R3: EV3: claim consume touches @RING, which the 'uart_rx' handler writes, outside a masked window -- mask the source around it or make the touch a Lane.A atomic (the interrupted flow must order against the handler)}}
  bcir.claim @consume attributes {
    claim_id = 3 : i32, phase = @drain, op = "uart.consume", reads = [@RING],
    writes = [@TAIL], count = 1 : i64, lane = #bcir.lane<u>,
    stride_class = #bcir.stride_class<scalar>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1 step 1 }
  bcir.claim @fill attributes {
    claim_id = 101 : i32, phase = @isr, op = "uart.rx_fill", reads = [],
    writes = [@RING], count = 1 : i64, lane = #bcir.lane<u>,
    stride_class = #bcir.stride_class<scalar>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<barriered>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1 step 1 }
}
