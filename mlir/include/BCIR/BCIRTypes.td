//===- BCIRTypes.td - BCIR type definitions -------------------*- tablegen -*-===//
#ifndef BCIR_TYPES
#define BCIR_TYPES

include "BCIR/BCIRDialect.td"

def BCIR_ResourceType : BCIR_Type<"Resource", "resource"> {
  let summary = "Registry-governed resource handle (RID-addressed, not a pointer)";
}

def BCIR_PathType : BCIR_Type<"Path", "path"> {
  let summary = "K_BCIR realization path handle";
}

def BCIR_StreamType : BCIR_Type<"Stream", "stream"> {
  let summary = "GEM StreamPack handle";
}

#endif // BCIR_TYPES
