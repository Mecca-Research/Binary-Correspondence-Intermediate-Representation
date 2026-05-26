#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace bcir {

enum class BcirLane : std::uint8_t { U = 0, UX = 1, T = 2, GGG = 3, A = 4, H = 5 };
enum class BcirOpcode : std::uint8_t {
  Load = 0, Store = 1, Add = 2, Sub = 3, Mul = 4, BitAnd = 5, BitOr = 6, BitXor = 7,
  AtomicAdd = 8, AtomicSub = 9, AtomicXor = 10, CmpXchg = 11, Barrier = 12, Phase = 13
};
enum class BcirEdgeKind : std::uint8_t { DataFlow = 0, ControlFlow = 1, HazardOrder = 2, PhaseOrder = 3, CostFlow = 4 };
enum class BcirHazardKind : std::uint8_t { RAW = 0, WAR = 1, WAW = 2 };
enum class BcirHazardMode : std::uint8_t { Unique = 0, Atomic = 1, Barriered = 2, Relaxed = 3 };
enum class BcirPhaseOrder : std::uint8_t { Monotonic = 0, EpochReset = 1 };
enum class BcirAddressSpace : std::uint8_t { Generic = 0, Global = 1, Shared = 2, Local = 3, Constant = 4, MMIO = 5 };
enum class BcirValueType : std::uint8_t { I32 = 0, I64 = 1, F32 = 2, F64 = 3 };
enum class BcirHazardDomain : std::uint8_t { RAM = 0, VRAM = 1, NVM = 2, MMIO = 3 };

enum class BcirRegistryFlag : std::uint32_t {
  None = 0u,
  Readable = 1u << 0u,
  Writable = 1u << 1u,
  AtomicCapable = 1u << 2u,
};

inline BcirRegistryFlag operator|(BcirRegistryFlag a, BcirRegistryFlag b) {
  return static_cast<BcirRegistryFlag>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

struct BcirCostTuple { double eSwitch = 0.0, tLatency = 0.0, mMove = 0.0, bPressure = 0.0, rRecompile = 0.0, sSync = 0.0, phiThermal = 0.0; };

struct BcirThetaHeader {
  std::uint32_t pointerSizeBytes = 8;
  std::uint32_t cacheLineBytes = 64;
  std::uint32_t vectorBits = 128;
  std::uint32_t warpWidth = 32;
};

struct BcirRegistryDescriptor {
  std::string rid;
  BcirLane laneType = BcirLane::U;
  BcirValueType elementType = BcirValueType::I32;
  std::uint32_t capacityBytes = 0;
  BcirAddressSpace addressSpace = BcirAddressSpace::Global;
  std::uint32_t aliasGroup = 0;
  std::uint32_t alignmentBytes = 4;
  std::uint32_t lifetimeBegin = 0;
  std::uint32_t lifetimeEnd = 0;
  BcirRegistryFlag flags = BcirRegistryFlag::Readable | BcirRegistryFlag::Writable;
enum class BcirContractMode : std::uint8_t { PromiseNoHazard = 0, RequiresOrdering = 1 };
enum class BcirAddressSpace : std::uint8_t { Generic = 0, Global = 1, Shared = 2, Local = 3, Constant = 4 };
enum class BcirLifetime : std::uint8_t { Temporary = 0, Persistent = 1, External = 2 };
enum class BcirOpcode : std::uint16_t { Unknown = 0, Load, Store, Add, Sub, Mul, And, Or, Xor, Barrier, AtomicAdd, AtomicSub, AtomicXor, CmpXchg, Phase };

struct BcirType {
  std::string scalar = "i32";
  std::uint16_t lanes = 1;
};

struct BcirRegistryDescriptor {
  std::string id;
  BcirAddressSpace addressSpace = BcirAddressSpace::Generic;
  std::uint64_t capacityBytes = 0;
  std::uint32_t alignmentBytes = 1;
  std::uint32_t aliasGroup = 0;
  bool supportsAtomic = false;
};

struct BcirCostTuple { double eSwitch = 0.0, tLatency = 0.0, mMove = 0.0, bPressure = 0.0, rRecompile = 0.0, sSync = 0.0, phiThermal = 0.0; };

struct BcirEdge { std::uint64_t src = 0, dst = 0; BcirEdgeKind kind = BcirEdgeKind::DataFlow; };

struct BcirNode {
  std::uint64_t id = 0;
  BcirOpcode opcode = BcirOpcode::Add;
  BcirValueType type = BcirValueType::I32;
  BcirOpcode opcode = BcirOpcode::Unknown;
  std::string opcodeText;
  BcirLane lane = BcirLane::U;
  BcirType type;
  std::uint32_t epoch = 0;
  std::uint32_t phase = 0;
  std::string rid;
  std::uint32_t offsetBytes = 0;
  std::vector<std::uint64_t> operands;
  BcirHazardMode hazardMode = BcirHazardMode::Relaxed;
  BcirHazardDomain hazardDomain = BcirHazardDomain::RAM;
  BcirCostTuple cost;
};

struct BcirEdge {
  std::uint64_t src = 0;
  std::uint64_t dst = 0;
  BcirEdgeKind kind = BcirEdgeKind::DataFlow;
  BcirHazardKind hazard = BcirHazardKind::RAW;
};

struct BcirGraph {
  BcirThetaHeader theta;
  std::vector<BcirRegistryDescriptor> registries;
  std::vector<BcirNode> nodes;
  std::vector<BcirEdge> edges;
};

struct BcirSchedule { std::vector<std::uint64_t> executionOrder; std::vector<std::size_t> phaseBegin; std::vector<std::size_t> phaseEnd; };
  std::optional<std::string> registry;
  std::uint64_t registryOffset = 0;
  std::vector<std::uint64_t> operands;
  std::vector<std::uint32_t> readsAliasGroups;
  std::vector<std::uint32_t> writesAliasGroups;
  std::uint64_t hazardDomain = 0;
  BcirCostTuple cost;
};

struct BcirGraph {
  std::vector<BcirNode> nodes;
  std::vector<BcirEdge> edges;
  std::unordered_map<std::string, BcirRegistryDescriptor> registries;
};

struct BcirSchedule {
  std::vector<std::uint64_t> executionOrder;
  std::vector<std::size_t> phaseBegin;
  std::vector<std::size_t> phaseEnd;
};

#pragma pack(push, 1)
struct BcirClaimV1 {
  std::uint8_t opcode = 0;
  std::uint8_t lane = 0;
  std::uint16_t phase = 0;
  std::uint16_t epoch = 0;
  std::uint16_t flags = 0;
  std::uint32_t strideBytes = 0;
  std::uint32_t rdRids[2] = {0, 0};
  std::uint32_t wrRids[2] = {0, 0};
  std::uint64_t hazardDomain = 0;
  std::uint64_t immediates[1] = {0};
  std::uint64_t immediates[2] = {0, 0};
  std::uint64_t costHint = 0;
  std::uint32_t reserved = 0;
};
#pragma pack(pop)
static_assert(sizeof(BcirClaimV1) == 64, "BcirClaimV1 must remain cache-line sized");

}  // namespace bcir
