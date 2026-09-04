//===- BCIREnums.h - BCIR enum declarations ----------------------*- C++ -*-===//
#ifndef BCIR_BCIRENUMS_H
#define BCIR_BCIRENUMS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

#include "BCIREnums.h.inc" // generated: -gen-enum-decls

namespace bcir {

// --- the derived decomposition of an ASN.1 transfer syntax -------------------------------
//
// BCIR_Asn1Rules stores ONE thing: which transfer syntax. Family, profile, canonicality
// and PER alignment are all functions of it, and they are computed here rather than
// stored alongside it so that no pair of attributes can ever disagree. R24's laws are
// written over these predicates, which is what lets "BCIR emits DER only" become "BCIR
// emits a canonical transfer syntax only" without the law having to enumerate anything.

// The Recommendation that defines the syntax, as its X-series number ("X.690", ...).
::llvm::StringRef asn1FamilyOf(Asn1Rules rules);

// Whether the octets are a FUNCTION of the abstract value.
//
// This is the property BCIR needs and the only one worth a law. A digest over a
// non-canonical encoding does not identify the value: the sender picks the spelling, so
// the sender picks the digest. X.690 9.1 makes the indefinite length form mandatory for
// constructed CER encodings, which is why `cer` is NOT canonical in this sense despite
// its name -- canonical there is about a canonical CHOICE among BER's options, not about
// byte-stability of a complete encoding.
bool isCanonicalAsn1Rules(Asn1Rules rules);

// Whether the syntax is one of X.691's ALIGNED variants. Meaningless -- and false --
// outside the PER family, which is why it is a question asked of the syntax rather than a
// separate attribute somebody could set on a JER operation.
bool asn1RulesAreAligned(Asn1Rules rules);

} // namespace bcir

#endif // BCIR_BCIRENUMS_H
