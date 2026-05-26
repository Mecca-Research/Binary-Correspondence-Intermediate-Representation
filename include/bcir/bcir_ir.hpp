#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace bcir {

enum class BcirLane : std::uint8_t { U = 0, UX = 1, T = 2, GGG = 3, A = 4, H = 5 };
enum class BcirEdgeKind : std::uint8_t { DataFlow = 0, ControlFlow = 1, HazardOrder = 2, PhaseOrder = 3, CostFlow = 4 };
enum class BcirHazardKind : std::uint8_t { RAW = 0, WAR = 1, WAW = 2 };
enum class BcirContractMode : std::uint8_t { PromiseNoHazard = 0, RequiresOrdering = 1 };

struct BcirCostTuple {
  double eSwitch = 0.0;
  double tLatency = 0.0;
  double mMove = 0.0;
  double bPressure = 0.0;
  double rRecompile = 0.0;
  double sSync = 0.0;
  double phiThermal = 0.0;
};

struct BcirEdge {
  std::uint64_t src = 0;
  std::uint64_t dst = 0;
  BcirEdgeKind kind = BcirEdgeKind::DataFlow;
};

struct BcirNode {
  std::uint64_t id = 0;
  std::string opcode;
  BcirLane lane = BcirLane::U;
  std::uint32_t epoch = 0;
  std::uint32_t phase = 0;
  std::optional<std::string> registry;
  std::optional<std::string> offsetExpr;
  std::vector<std::uint64_t> operands;
  BcirCostTuple cost;
};

#pragma pack(push, 1)
struct BcirClaimV1 {
  std::uint8_t opcode = 0;
  std::uint8_t lane = 0;
  std::uint16_t phase = 0;
  std::uint16_t epoch = 0;
  std::uint16_t flags = 0;
  std::uint32_t strideBytes = 0;
  std::uint32_t rdRids[4] = {0, 0, 0, 0};
  std::uint32_t wrRids[4] = {0, 0, 0, 0};
  std::uint64_t hazardDomain = 0;
  std::uint64_t immediates[2] = {0, 0};
  std::uint64_t costHint = 0;
};
#pragma pack(pop)

static_assert(sizeof(BcirClaimV1) == 64, "BcirClaimV1 must remain cache-line sized");

}  // namespace bcir
