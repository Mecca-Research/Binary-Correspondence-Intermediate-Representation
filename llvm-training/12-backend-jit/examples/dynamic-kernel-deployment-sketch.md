# Dynamic BCIR kernel deployment sketch

The control plane below combines lazy compilation, speculation, hot re-JIT,
symbol replacement, and resource retirement.

```text
JITDylibs
  bcir.runtime       stable ABI + telemetry + target adapters
  bcir.entry         public bcir.kernel.<id> stubs/reexports
  bcir.impl.g7       active generation, tracker RT7
  bcir.speculative   unpublished candidates, tracker per candidate

onGraphAccepted(graph):
  validateBCIRSemantics(graph)
  recipe = lowerMLIRToLLVMRecipe(graph, selectedTarget)
  defineLazyEntry("bcir.kernel.42", recipe)

onFirstCall("bcir.kernel.42"):
  candidate = compile(recipe, baselinePipeline, generation=7, tracker=RT7)
  validateLLVMABIAndArtifact(candidate)
  publishStableEntry("bcir.kernel.42", candidate.symbol)
  telemetry.recordPublication(graph=42, generation=7, target, pipeline)

onLikelyNextKernel(graph=43):
  RTS = createResourceTracker("speculative-43")
  queueLowPriorityCompile(graph=43, tracker=RTS, token=policyEpoch)
  if policyEpoch changed before commit:
    RTS.remove()

onHot(graph=42, expectedGeneration=7, profileSnapshot=P19):
  RT8 = createResourceTracker("kernel-42-g8")
  candidate = compile(recipe, profilePipeline(P19), generation=8, tracker=RT8)
  validateBCIRConformance(candidate)

  lock publication for graph 42
  if activeGeneration != 7:
    unlock; RT8.remove(); return LOST_RACE
  atomicallyRedirectStableEntry("bcir.kernel.42", candidate.symbol)
  activeGeneration = 8
  unlock

  waitForQuiescence(generation=7)
  telemetry.detachAddressRecords(generation=7)
  RT7.remove()
```

The stable entry name is the client ABI. Generation-specific implementation
symbols are private and may disappear. The publication lock can be replaced by a
compare-and-swap generation protocol, but compilation completion alone must
never decide the winner.
