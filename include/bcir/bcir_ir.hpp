#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace bcir {

enum class BcirLane : std::uint8_t { U = 0, UX = 1, T = 2, GGG = 3, A = 4, H = 5 };
enum class BcirEdgeKind : std::uint8_t { DataFlow = 0, ControlFlow = 1, HazardOrder = 2, PhaseOrder = 3, CostFlow = 4 };
enum class BcirHazardKind : std::uint8_t { RAW = 0, WAR = 1, WAW = 2 };
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
  BcirOpcode opcode = BcirOpcode::Unknown;
  std::string opcodeText;
  BcirLane lane = BcirLane::U;
  BcirType type;
  std::uint32_t epoch = 0;
  std::uint32_t phase = 0;
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
  std::uint64_t immediates[2] = {0, 0};
  std::uint64_t costHint = 0;
  std::uint32_t reserved = 0;
};
#pragma pack(pop)

static_assert(sizeof(BcirClaimV1) == 64, "BcirClaimV1 must remain cache-line sized");

}  // namespace bcir
