//===- BCIRParseOps.td - grammar / parser ops (M5) ----------*- tablegen -*-===//
//
// Grammar -> tokens -> BCIR claims. Mirrors bcir/etl/parse.py. A grammar lowers
// to a transducer machine (parse.lower_to_fsm): parser generator = grammar ->
// event machine.
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_PARSE_OPS
#define BCIR_PARSE_OPS

include "BCIR/BCIRDialect.td"
include "mlir/IR/SymbolInterfaces.td"

def BCIR_ParseGrammarOp : BCIR_Op<"parse.grammar", [Symbol, SymbolTable, NoTerminator]> {
  let summary = "Structured-text grammar container";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$syntax, StrAttr:$start_symbol);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def BCIR_ParseTokenOp : BCIR_Op<"parse.token", [Symbol]> {
  let summary = "Token definition (regex pattern, skip channel, precedence)";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$pattern,
                       DefaultValuedAttr<BoolAttr, "false">:$skip,
                       DefaultValuedAttr<I32Attr, "0">:$precedence);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def BCIR_ParseRuleOp : BCIR_Op<"parse.rule", [Symbol]> {
  let summary = "Grammar production rule";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$lhs, ArrayAttr:$rhs, StrAttr:$action);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_ParseLowerToFSMOp : BCIR_Op<"parse.lower_to_fsm", [Symbol]> {
  let summary = "Grammar-to-transducer lowering contract";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$grammar,
                       FlatSymbolRefAttr:$machine, StrAttr:$algorithm, StrAttr:$conflict_policy);
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // BCIR_PARSE_OPS
