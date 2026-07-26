"""ASN.1 (X.680) front-end: module text in, `bcir.asn1.schema` model out.

The rest of the ASN.1 rail (`bcir/asn1/`) is the ENCODER: given a type model it
produces and checks X.690 octets. This package is the COMPILER FRONT-END that builds
that model from an actual `.asn1` module, so a peer's schema can be consumed instead
of hand-transcribed into Python.

    from bcir.frontends.asn1 import compile_module
    lowered = compile_module(open("Peer.asn1").read())
    octets  = lowered.module.encode("SubjectPublicKeyInfo", value)
"""

from .lexer import Asn1SyntaxError, tokenize
from .lower import Asn1SemanticError, LoweredModule, compile_module, lower
from .parser import parse_module, parse_modules
from .printer import print_module

__all__ = ["Asn1SemanticError", "Asn1SyntaxError", "LoweredModule", "compile_module",
           "lower", "parse_module", "parse_modules", "print_module", "tokenize"]
