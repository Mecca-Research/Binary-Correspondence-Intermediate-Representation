//===- BCIROps.td - BCIR dialect-family umbrella ------------*- tablegen -*-===//
//
// One TableGen entry point for op generation. The family shares the `bcir`
// namespace during bootstrap; conceptually each include is a distinct layer of
// law (LangRef Sec. 2).
//
//   BCIRCoreOps             module/registry/resource/phase/claim/load/store/...  (BCIR-0..2)
//   BCIRTargetOps           target.capability                                    (H container)
//   BCIRMemOps              mem.tier/mem.ham/mem.cxl_swap                         (CT1)
//   BCIRKBCIROps            kbcir.policy/path/plan/select                         (BCIR-3)
//   BCIRGEMOps              gem.stream_pack/lane_segment/prefetch/block           (BCIR-4)
//   BCIRTraceOps            trace.note                                           (cross-cut)
//   BCIRVerifyOps           verify.* (R1-R12 as IR)                              (M1)
//   BCIROptOps              opt.* (rewrite/layout/mem laws as IR)                (M2)
//   BCIRLoweringContractOps isa.*/packet.*/target.lower_contract                 (M3, BCIR-5)
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_OPS
#define BCIR_OPS

include "BCIR/BCIRCoreOps.td"
include "BCIR/BCIRTargetOps.td"
include "BCIR/BCIRMemOps.td"
include "BCIR/BCIRKBCIROps.td"
include "BCIR/BCIRGEMOps.td"
include "BCIR/BCIRTraceOps.td"
include "BCIR/BCIRVerifyOps.td"
include "BCIR/BCIROptOps.td"
include "BCIR/BCIRLoweringContractOps.td"

// M5 Event Transduction Layer (GEM-E): the mouth of BCIR.
include "BCIR/BCIREventOps.td"
include "BCIR/BCIRTransducerOps.td"
include "BCIR/BCIRParseOps.td"
include "BCIR/BCIRBinaryFormatOps.td"

// Phase 8: async dependency tokens.
include "BCIR/BCIRAsyncOps.td"

#endif // BCIR_OPS
