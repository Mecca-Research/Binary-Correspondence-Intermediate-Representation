// RUN: bcir-opt %s | FileCheck %s
//
// M5: an NVMe SQE command header as a BCIR binary format. The oracle
// bcir/etl/binary.py decodes the same record today:
//   python -c "from bcir.etl import decode; from bcir.etl.binary import NVME_SQE_HEADER; \
//     print(decode(NVME_SQE_HEADER, bytes([1,0,0x34,0x12,0xDD,0xCC,0xBB,0xAA])))"
// -> {'opcode': 1, 'command_id': 4660, 'namespace_id': 2864434397}
// NOT parseable on this host (no bcir-opt).

bcir.module @nvme_packet_decode_skeleton attributes {
  version = "0.1", target_model = "registry-first", execution_model = "gem-e", cost_model = "k_bcir"
} {
  bcir.event.stream @nvme_bytes { kind = "binary", encoding = "le", element_bits = 8 : i64, max_window = 64 : i64 }

  bcir.binary.format @nvme_cmd attributes { endianness = "little", alignment_bits = 32 : i64 } {
    bcir.binary.field @opcode { name = "opcode", offset_bits = 0 : i64, width_bits = 8 : i64, kind = "u", semantic = "command_opcode" }
    bcir.binary.field @cid    { name = "command_id", offset_bits = 16 : i64, width_bits = 16 : i64, kind = "u", semantic = "queue_command_id" }
    bcir.binary.field @nsid   { name = "namespace_id", offset_bits = 32 : i64, width_bits = 32 : i64, kind = "u", semantic = "resource_namespace" }
    bcir.binary.record @sqe_header { name = "nvme_sqe_header", fields = [@opcode, @cid, @nsid], size_bits = 64 : i64 }
  }

  bcir.binary.decode @decode_sqe { format = @nvme_cmd, stream = @nvme_bytes, emits = "event.nvme_command" }
}

// CHECK-LABEL: bcir.module @nvme_packet_decode_skeleton
// CHECK: bcir.event.stream @nvme_bytes
// CHECK: bcir.binary.format @nvme_cmd
// CHECK: bcir.binary.record @sqe_header
// CHECK: bcir.binary.decode @decode_sqe
