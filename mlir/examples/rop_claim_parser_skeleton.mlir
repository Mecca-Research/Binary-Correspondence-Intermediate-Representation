// RUN: bcir-opt %s | FileCheck %s
//
// M5: a mini ROP-claim parser expressed as BCIR IR -- grammar + transducer +
// lowering. The oracle bcir/etl/parse.py runs the same shape today:
//   python -c "from bcir.etl import parse_rop; print(parse_rop('claim add { reads A,B; writes C; count 1024; lane u; stride unit }'))"
// NOT parseable on this host (no bcir-opt).

bcir.module @rop_claim_parser_skeleton attributes {
  version = "0.1", target_model = "registry-first", execution_model = "gem-e", cost_model = "k_bcir"
} {
  bcir.event.stream @source_text { kind = "text", encoding = "utf8", element_bits = 8 : i64, max_window = 8192 : i64 }

  bcir.parse.grammar @rop_claim_grammar attributes { syntax = "ebnf", start_symbol = "claim" } {
    bcir.parse.token @IDENT { pattern = "[A-Za-z_][A-Za-z0-9_]*", skip = false, precedence = 0 : i32 }
    bcir.parse.token @INT   { pattern = "[0-9]+", skip = false, precedence = 0 : i32 }
    bcir.parse.token @WS    { pattern = "[ \\t\\r\\n]+", skip = true, precedence = 0 : i32 }
    bcir.parse.rule @claim_rule { lhs = "claim", rhs = ["claim", "IDENT", "{", "body", "}"], action = "emit_bcir_claim" }
  }

  bcir.fsm.machine @rop_claim_machine attributes { kind = "transducer", input_stream = @source_text, start_state = @s0 } {
    bcir.fsm.state @s0 { accepting = false, error = false }
    bcir.fsm.state @s_accept { accepting = true, error = false }
    bcir.fsm.transition @t_claim { from = @s0, to = @s_accept, on = "token.claim", guard = "next_token_is_IDENT", action = "capture_claim_name" }
  }

  bcir.parse.lower_to_fsm @lower_rop_claim { grammar = @rop_claim_grammar, machine = @rop_claim_machine, algorithm = "lalr", conflict_policy = "error" }
}

// CHECK-LABEL: bcir.module @rop_claim_parser_skeleton
// CHECK: bcir.parse.grammar @rop_claim_grammar
// CHECK: bcir.fsm.machine @rop_claim_machine
// CHECK: bcir.parse.lower_to_fsm @lower_rop_claim
