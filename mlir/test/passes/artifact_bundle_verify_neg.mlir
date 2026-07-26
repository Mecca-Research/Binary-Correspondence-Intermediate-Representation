// RUN: bcir-opt -verify-diagnostics -split-input-file %s

// expected-error @+1 {{unknown artifact kind 'mystery'}}
bcir.artifact.variant @bad_kind {
  kind = "mystery", format = "text", triple = "", architecture = "", os_abi = "",
  channel = "", entry_symbol = "", endianness = "neutral", pointer_bits = 0 : i32,
  machine = 0 : i64, priority = 0 : i32, flags = 0 : i32,
  provenance_digest = "0000000000000000", required_features = [],
  prohibited_features = [], payload_offset = 128 : i64, payload_size = 1 : i64,
  payload_crc32 = 0 : i64,
  payload_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  target_manifest_sha256 = "", cal_gen = 0 : i64
}

// -----

// expected-error @+1 {{feature arrays must contain strictly sorted unique strings}}
bcir.artifact.variant @bad_features {
  kind = "c_source", format = "text", triple = "", architecture = "", os_abi = "",
  channel = "", entry_symbol = "", endianness = "neutral", pointer_bits = 0 : i32,
  machine = 0 : i64, priority = 0 : i32, flags = 0 : i32,
  provenance_digest = "0000000000000000",
  required_features = ["z", "a"], prohibited_features = [],
  payload_offset = 128 : i64, payload_size = 1 : i64, payload_crc32 = 0 : i64,
  payload_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  target_manifest_sha256 = "", cal_gen = 0 : i64
}

// -----

// expected-error @+1 {{classification must be exact|quantized|approximate}}
bcir.artifact.selection @bad_selection {
  bundle = @bundle, variant = "x", classification = "maybe",
  envelope_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  generation = 0 : i64
}

// -----

// expected-error @+1 {{native object metadata requires pointer_bits, endianness, and machine}}
bcir.artifact.variant @native_without_machine {
  kind = "elf_object", format = "elf", triple = "", architecture = "x86_64",
  os_abi = "linux-gnu", channel = "host", entry_symbol = "bcir_kernel",
  endianness = "little", pointer_bits = 64 : i32, machine = 0 : i64,
  priority = 0 : i32, flags = 3 : i32,
  provenance_digest = "0000000000000000", required_features = [],
  prohibited_features = [], payload_offset = 128 : i64, payload_size = 20 : i64,
  payload_crc32 = 0 : i64,
  payload_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  target_manifest_sha256 = "", cal_gen = 0 : i64
}

// -----

// expected-error @+1 {{named executable artifact kinds require the executable flag}}
bcir.artifact.variant @unmarked_executable {
  kind = "elf_executable", format = "elf", triple = "", architecture = "x86_64",
  os_abi = "linux-gnu", channel = "host", entry_symbol = "main",
  endianness = "little", pointer_bits = 64 : i32, machine = 62 : i64,
  priority = 0 : i32, flags = 1 : i32,
  provenance_digest = "0000000000000000", required_features = [],
  prohibited_features = [], payload_offset = 128 : i64, payload_size = 20 : i64,
  payload_crc32 = 0 : i64,
  payload_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  target_manifest_sha256 = "", cal_gen = 0 : i64
}

// -----

// expected-error @+1 {{root_variant must name a stream_pack}}
bcir.artifact.bundle @bad_root attributes {
  version = 1 : i32, root_variant = "source", default_variant = "source",
  provenance_digest = "0000000000000000", generation = 0 : i64, wire_bytes = 584 : i64,
  artifact_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  body_crc32 = 0 : i64, header_crc32 = 0 : i64
} {
  bcir.artifact.variant @source {
    kind = "c_source", format = "text", triple = "", architecture = "", os_abi = "",
    channel = "", entry_symbol = "", endianness = "neutral", pointer_bits = 0 : i32,
    machine = 0 : i64, priority = 0 : i32, flags = 0 : i32,
    provenance_digest = "0000000000000000", required_features = [],
    prohibited_features = [], payload_offset = 576 : i64, payload_size = 1 : i64,
    payload_crc32 = 0 : i64,
    payload_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    target_manifest_sha256 = "", cal_gen = 0 : i64
  }
}

// -----

// expected-error @+1 {{executable default_variant must carry R12 attestation}}
bcir.artifact.bundle @bad_default attributes {
  version = 1 : i32, root_variant = "", default_variant = "run",
  provenance_digest = "0000000000000000", generation = 0 : i64, wire_bytes = 600 : i64,
  artifact_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  body_crc32 = 0 : i64, header_crc32 = 0 : i64
} {
  bcir.artifact.variant @run {
    kind = "elf_executable", format = "elf", triple = "", architecture = "x86_64",
    os_abi = "linux-gnu", channel = "host", entry_symbol = "main",
    endianness = "little", pointer_bits = 64 : i32, machine = 62 : i64,
    priority = 0 : i32, flags = 2 : i32,
    provenance_digest = "0000000000000000", required_features = [],
    prohibited_features = [], payload_offset = 576 : i64, payload_size = 20 : i64,
    payload_crc32 = 0 : i64,
    payload_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    target_manifest_sha256 = "", cal_gen = 0 : i64
  }
}

// -----

// expected-error @+1 {{feature arrays must contain strictly sorted unique strings}}
bcir.artifact.variant @bad_feature_spelling {
  kind = "c_source", format = "text", triple = "", architecture = "", os_abi = "",
  channel = "", entry_symbol = "", endianness = "neutral", pointer_bits = 0 : i32,
  machine = 0 : i64, priority = 0 : i32, flags = 0 : i32,
  provenance_digest = "0000000000000000",
  required_features = ["bad feature"], prohibited_features = [],
  payload_offset = 128 : i64, payload_size = 1 : i64, payload_crc32 = 0 : i64,
  payload_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  target_manifest_sha256 = "", cal_gen = 0 : i64
}

// -----

// expected-error @+1 {{provenance_digest must be 16 lowercase hexadecimal digits}}
bcir.artifact.variant @bad_provenance {
  kind = "c_source", format = "text", triple = "", architecture = "", os_abi = "",
  channel = "", entry_symbol = "", endianness = "neutral", pointer_bits = 0 : i32,
  machine = 0 : i64, priority = 0 : i32, flags = 0 : i32,
  provenance_digest = "ABC",
  required_features = [], prohibited_features = [],
  payload_offset = 128 : i64, payload_size = 1 : i64, payload_crc32 = 0 : i64,
  payload_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  target_manifest_sha256 = "", cal_gen = 0 : i64
}

// -----

// expected-error @+1 {{bundle symbol 'missing' does not resolve to bcir.artifact.bundle}}
bcir.artifact.selection @missing_bundle {
  bundle = @missing, variant = "source", classification = "exact",
  envelope_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  generation = 0 : i64
}

// -----

bcir.artifact.bundle @selection_bundle attributes {
  version = 1 : i32, root_variant = "", default_variant = "source",
  provenance_digest = "0000000000000000", generation = 4 : i64, wire_bytes = 584 : i64,
  artifact_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  body_crc32 = 0 : i64, header_crc32 = 0 : i64
} {
  bcir.artifact.variant @source {
    kind = "c_source", format = "text", triple = "", architecture = "", os_abi = "",
    channel = "", entry_symbol = "", endianness = "neutral", pointer_bits = 0 : i32,
    machine = 0 : i64, priority = 0 : i32, flags = 0 : i32,
    provenance_digest = "0000000000000000",
    required_features = [], prohibited_features = [],
    payload_offset = 576 : i64, payload_size = 1 : i64, payload_crc32 = 0 : i64,
    payload_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    target_manifest_sha256 = "", cal_gen = 0 : i64
  }
}

// expected-error @+1 {{generation does not match bundle generation 4}}
bcir.artifact.selection @stale_selection {
  bundle = @selection_bundle, variant = "source", classification = "exact",
  envelope_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  generation = 3 : i64
}

// -----

bcir.artifact.bundle @selection_bundle attributes {
  version = 1 : i32, root_variant = "", default_variant = "source",
  provenance_digest = "0000000000000000", generation = 4 : i64, wire_bytes = 584 : i64,
  artifact_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  body_crc32 = 0 : i64, header_crc32 = 0 : i64
} {
  bcir.artifact.variant @source {
    kind = "c_source", format = "text", triple = "", architecture = "", os_abi = "",
    channel = "", entry_symbol = "", endianness = "neutral", pointer_bits = 0 : i32,
    machine = 0 : i64, priority = 0 : i32, flags = 0 : i32,
    provenance_digest = "0000000000000000",
    required_features = [], prohibited_features = [],
    payload_offset = 576 : i64, payload_size = 1 : i64, payload_crc32 = 0 : i64,
    payload_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    target_manifest_sha256 = "", cal_gen = 0 : i64
  }
}

// expected-error @+1 {{variant 'absent' is not present in bundle 'selection_bundle'}}
bcir.artifact.selection @missing_variant {
  bundle = @selection_bundle, variant = "absent", classification = "exact",
  envelope_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  generation = 4 : i64
}
