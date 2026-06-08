//===- BCIRAsyncOps.td - async dependency tokens (Phase 8) --*- tablegen -*-===//
//
// Makes BCIR concurrency explicit and verifiable: `bcir.async.fork` launches a
// claim asynchronously and yields a `!bcir.token`; `bcir.async.await` joins a set
// of tokens. The phase DAG gives coarse ordering; tokens give the fine-grained,
// SSA-explicit async-dependency edges (mirrors bcir/gem/async_tokens.py). They
// lower (future) to MLIR `async`/coroutines or a runtime scheduler.
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_ASYNC_OPS
#define BCIR_ASYNC_OPS

include "BCIR/BCIRDialect.td"
include "BCIR/BCIRTypes.td"
include "mlir/IR/SymbolInterfaces.td"

def BCIR_AsyncForkOp : BCIR_Op<"async.fork"> {
  let summary = "Launch a claim asynchronously, yielding a completion token";
  let arguments = (ins FlatSymbolRefAttr:$claim);
  let results = (outs BCIR_TokenType:$token);
  let assemblyFormat = "$claim attr-dict `:` type($token)";
}

def BCIR_AsyncAwaitOp : BCIR_Op<"async.await"> {
  let summary = "Await (join) a set of async completion tokens";
  let arguments = (ins Variadic<BCIR_TokenType>:$tokens);
  let assemblyFormat = "$tokens attr-dict `:` type($tokens)";
}

#endif // BCIR_ASYNC_OPS
