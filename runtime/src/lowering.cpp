#include "bcir/lowering.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

namespace bcir {
namespace {

Diagnostic make_diag(const SourceLocation& loc, std::string pass, std::string code,
                     std::string message) {
  Diagnostic d;
  d.location = loc;
  d.pass = std::move(pass);
  d.code = std::move(code);
  d.message = std::move(message);
  return d;
}

bool is_register(std::string_view text) {
  static const std::regex kRidRegex(R"(^r[0-9]+$)");
  return std::regex_match(text.begin(), text.end(), kRidRegex);
}

int parse_rid_number(const std::string& rid) {
  if (!is_register(rid)) {
    return -1;
  }
  return std::stoi(rid.substr(1));
}

std::optional<int> parse_numeric_suffix(const std::string& text) {
  std::size_t split = text.size();
  while (split > 0 &&
         std::isdigit(static_cast<unsigned char>(text[split - 1]))) {
    --split;
  }
  if (split == text.size()) {
    return std::nullopt;
  }
  return std::stoi(text.substr(split));
}

std::set<std::string> collect_block_symbols(const BlockNode& block) {
  std::set<std::string> symbols;
  for (const auto& operation : block.operations) {
    if (operation->kind == Operation::Kind::LdSt) {
      const auto* op = static_cast<const LdStOperation*>(operation.get());
      symbols.insert(op->rid);
    } else if (operation->kind == Operation::Kind::Binary) {
      const auto* op = static_cast<const BinaryOpOperation*>(operation.get());
      symbols.insert(op->dst);
      symbols.insert(op->lhs);
      symbols.insert(op->rhs);
    } else if (operation->kind == Operation::Kind::Lane) {
      const auto* op = static_cast<const LaneOpOperation*>(operation.get());
      symbols.insert(op->rid);
    } else if (operation->kind == Operation::Kind::MapSurface) {
      const auto* op = static_cast<const MapSurfaceOperation*>(operation.get());
      symbols.insert(op->rid);
    }
  }
  return symbols;
}

std::unique_ptr<Operation> clone_operation(const Operation& operation) {
  switch (operation.kind) {
    case Operation::Kind::LdSt: {
      const auto& src = static_cast<const LdStOperation&>(operation);
      auto out = std::make_unique<LdStOperation>(src.isLoad, src.location);
      out->rid = src.rid;
      out->registryType = src.registryType;
      out->lane = src.lane;
      out->target = src.target;
      return out;
    }
    case Operation::Kind::Binary: {
      const auto& src = static_cast<const BinaryOpOperation&>(operation);
      auto out = std::make_unique<BinaryOpOperation>(src.location);
      out->opcode = src.opcode;
      out->dst = src.dst;
      out->dstType = src.dstType;
      out->lhs = src.lhs;
      out->lhsType = src.lhsType;
      out->rhs = src.rhs;
      out->rhsType = src.rhsType;
      return out;
    }
    case Operation::Kind::Lane: {
      const auto& src = static_cast<const LaneOpOperation&>(operation);
      auto out = std::make_unique<LaneOpOperation>(src.location);
      out->opcode = src.opcode;
      out->rid = src.rid;
      out->lane = src.lane;
      return out;
    }
    case Operation::Kind::Phase: {
      const auto& src = static_cast<const PhaseOperation&>(operation);
      auto out = std::make_unique<PhaseOperation>(src.location);
      out->phase = src.phase;
      return out;
    }
    case Operation::Kind::Barrier: {
      const auto& src = static_cast<const BarrierOperation&>(operation);
      auto out = std::make_unique<BarrierOperation>(src.location);
      out->scope = src.scope;
      return out;
    }
    case Operation::Kind::MacroExpansion: {
      const auto& src = static_cast<const MacroExpansionOperation&>(operation);
      auto out = std::make_unique<MacroExpansionOperation>(src.location);
      out->macroName = src.macroName;
      out->args = src.args;
      return out;
    }
    case Operation::Kind::MapSurface: {
      const auto& src = static_cast<const MapSurfaceOperation&>(operation);
      auto out = std::make_unique<MapSurfaceOperation>(src.location);
      out->surfaceKind = src.surfaceKind;
      out->rid = src.rid;
      out->lane = src.lane;
      out->target = src.target;
      return out;
    }
  }
  return nullptr;
}

void replace_symbol(std::string* symbol, const std::map<std::string, std::string>& binding,
                    const std::map<std::string, std::string>& temp_map) {
  const auto bind_it = binding.find(*symbol);
  if (bind_it != binding.end()) {
    *symbol = bind_it->second;
    return;
  }
  const auto temp_it = temp_map.find(*symbol);
  if (temp_it != temp_map.end()) {
    *symbol = temp_it->second;
  }
}

void apply_binding(Operation* operation, const std::map<std::string, std::string>& binding,
                   const std::map<std::string, std::string>& temp_map) {
  if (operation->kind == Operation::Kind::LdSt) {
    auto* op = static_cast<LdStOperation*>(operation);
    replace_symbol(&op->rid, binding, temp_map);
  } else if (operation->kind == Operation::Kind::Binary) {
    auto* op = static_cast<BinaryOpOperation*>(operation);
    replace_symbol(&op->dst, binding, temp_map);
    replace_symbol(&op->lhs, binding, temp_map);
    replace_symbol(&op->rhs, binding, temp_map);
  } else if (operation->kind == Operation::Kind::Lane) {
    auto* op = static_cast<LaneOpOperation*>(operation);
    replace_symbol(&op->rid, binding, temp_map);
  } else if (operation->kind == Operation::Kind::MapSurface) {
    auto* op = static_cast<MapSurfaceOperation*>(operation);
    replace_symbol(&op->rid, binding, temp_map);
  }
}

int find_next_register_id(const std::set<std::string>& symbols) {
  int max_id = -1;
  for (const std::string& symbol : symbols) {
    max_id = std::max(max_id, parse_rid_number(symbol));
  }
  return max_id + 1;
}

}  // namespace

std::string lowering_component_banner() {
  return "bcir-lowering: BCIR<->ROP<->LLVM conversions";
}

std::vector<Diagnostic> expand_macros(ModuleNode* module) {
  std::vector<Diagnostic> diags;
  if (module == nullptr) {
    return diags;
  }

  std::map<std::string, const MacroDefinitionNode*> macros;
  for (const auto& macro : module->macros) {
    macros.emplace(macro.name, &macro);
  }

  for (FunctionNode& fn : module->functions) {
    std::vector<std::unique_ptr<Operation>> rewritten;
    std::set<std::string> symbols = collect_block_symbols(fn.body);
    int next_rid = find_next_register_id(symbols);

    for (const auto& operation : fn.body.operations) {
      if (operation->kind != Operation::Kind::MacroExpansion) {
        rewritten.push_back(clone_operation(*operation));
        continue;
      }

      const auto* expand = static_cast<const MacroExpansionOperation*>(operation.get());
      const auto macro_it = macros.find(expand->macroName);
      if (macro_it == macros.end()) {
        diags.push_back(make_diag(expand->location, "macro_expansion", "macro.not_found",
                                  "unknown macro '" + expand->macroName + "'"));
        continue;
      }

      const MacroDefinitionNode& macro = *macro_it->second;
      if (macro.params.size() != expand->args.size()) {
        diags.push_back(make_diag(expand->location, "macro_expansion",
                                  "macro.arity_mismatch",
                                  "macro '" + macro.name + "' expects " +
                                      std::to_string(macro.params.size()) +
                                      " argument(s) but got " +
                                      std::to_string(expand->args.size())));
        continue;
      }

      std::map<std::string, std::string> binding;
      for (std::size_t i = 0; i < macro.params.size(); ++i) {
        binding.emplace(macro.params[i], expand->args[i]);
      }

      std::map<std::string, std::string> temp_map;
      std::set<std::string> macro_symbols = collect_block_symbols(macro.body);
      for (const std::string& symbol : macro_symbols) {
        const bool is_param = binding.find(symbol) != binding.end();
        if (is_param) {
          continue;
        }
        if (!is_register(symbol)) {
          continue;
        }

        if (symbols.find(symbol) != symbols.end()) {
          const std::string fresh = "r" + std::to_string(next_rid++);
          temp_map.emplace(symbol, fresh);
          symbols.insert(fresh);
        } else {
          symbols.insert(symbol);
        }
      }

      for (const auto& [param, arg] : binding) {
        if (temp_map.find(param) != temp_map.end()) {
          diags.push_back(make_diag(expand->location, "macro_expansion",
                                    "macro.hygiene.param_temp_collision",
                                    "parameter '" + param +
                                        "' collides with macro temporary"));
        }
        if (!arg.empty() && !std::isalnum(static_cast<unsigned char>(arg.front())) &&
            arg.front() != '_') {
          diags.push_back(make_diag(expand->location, "macro_expansion",
                                    "macro.arg.invalid_symbol",
                                    "argument '" + arg +
                                        "' is not a valid bindable symbol"));
        }
      }

      for (const auto& macro_op : macro.body.operations) {
        auto cloned = clone_operation(*macro_op);
        apply_binding(cloned.get(), binding, temp_map);
        cloned->location = expand->location;

        rewritten.push_back(std::move(cloned));
      }
    }

    fn.body.operations = std::move(rewritten);
  }

  return diags;
}

void lower_map_surface_ops(ModuleNode* module) {
  if (module == nullptr) {
    return;
  }

  for (FunctionNode& fn : module->functions) {
    std::vector<std::unique_ptr<Operation>> lowered;

    for (const auto& operation : fn.body.operations) {
      if (operation->kind != Operation::Kind::MapSurface) {
        lowered.push_back(clone_operation(*operation));
        continue;
      }

      const auto* map_op = static_cast<const MapSurfaceOperation*>(operation.get());
      if (map_op->surfaceKind == MapSurfaceOperation::SurfaceKind::Load) {
        auto ld = std::make_unique<LdStOperation>(true, map_op->location);
        ld->rid = map_op->rid;
        ld->lane = map_op->lane;
        ld->target = map_op->target;
        lowered.push_back(std::move(ld));
      } else if (map_op->surfaceKind == MapSurfaceOperation::SurfaceKind::Store) {
        auto st = std::make_unique<LdStOperation>(false, map_op->location);
        st->rid = map_op->rid;
        st->lane = map_op->lane;
        st->target = map_op->target;
        lowered.push_back(std::move(st));
      } else {
        auto begin_barrier = std::make_unique<BarrierOperation>(map_op->location);
        begin_barrier->scope = "map_atomic_begin";
        lowered.push_back(std::move(begin_barrier));

        auto ld = std::make_unique<LdStOperation>(true, map_op->location);
        ld->rid = map_op->rid;
        ld->lane = map_op->lane;
        ld->target = map_op->target;
        lowered.push_back(std::move(ld));

        auto bin = std::make_unique<BinaryOpOperation>(map_op->location);
        if (map_op->surfaceKind == MapSurfaceOperation::SurfaceKind::AtomicAdd) {
          bin->opcode = "add";
        } else if (map_op->surfaceKind == MapSurfaceOperation::SurfaceKind::AtomicSub) {
          bin->opcode = "sub";
        } else {
          bin->opcode = "xor";
        }
        bin->dst = map_op->rid;
        bin->lhs = map_op->rid;
        bin->rhs = map_op->rid;
        lowered.push_back(std::move(bin));

        auto st = std::make_unique<LdStOperation>(false, map_op->location);
        st->rid = map_op->rid;
        st->lane = map_op->lane;
        st->target = map_op->target;
        lowered.push_back(std::move(st));

        auto end_barrier = std::make_unique<BarrierOperation>(map_op->location);
        end_barrier->scope = "map_atomic_end";
        lowered.push_back(std::move(end_barrier));
      }
    }

    fn.body.operations = std::move(lowered);
  }
}

std::vector<RopInstruction> bcir_graph_to_rop_stream(const BlockNode& block) {
  std::vector<RopInstruction> stream;
  stream.reserve(block.operations.size());

  std::unordered_map<std::string, std::size_t> last_writer;
  std::optional<std::size_t> last_control_index;
  int current_phase = 0;

  auto track_source_dependency = [&](RopInstruction* instruction,
                                     const std::string& symbol) {
    if (symbol.empty()) {
      return;
    }
    const auto writer_it = last_writer.find(symbol);
    if (writer_it == last_writer.end()) {
      return;
    }
    instruction->dependencies.push_back(RopDependency{writer_it->second, "data"});
  };

  for (const auto& operation : block.operations) {
    RopInstruction instruction;

    if (operation->kind == Operation::Kind::LdSt) {
      const auto* op = static_cast<const LdStOperation*>(operation.get());
      instruction.opcode = op->isLoad ? "ld" : "st";
      instruction.rid = op->rid;
      instruction.laneType = op->registryType;
      instruction.lane = op->lane;
      instruction.offset = parse_numeric_suffix(op->target);
      instruction.target = op->target;
      instruction.isLoad = op->isLoad;
      track_source_dependency(&instruction, op->rid);
    } else if (operation->kind == Operation::Kind::Binary) {
      const auto* op = static_cast<const BinaryOpOperation*>(operation.get());
      instruction.opcode = op->opcode;
      instruction.rid = op->dst;
      instruction.laneType = op->dstType;
      instruction.lhs = op->lhs;
      instruction.rhs = op->rhs;
      track_source_dependency(&instruction, op->lhs);
      track_source_dependency(&instruction, op->rhs);
    } else if (operation->kind == Operation::Kind::Lane) {
      const auto* op = static_cast<const LaneOpOperation*>(operation.get());
      instruction.opcode = "lane." + op->opcode;
      instruction.rid = op->rid;
      instruction.lane = op->lane;
      track_source_dependency(&instruction, op->rid);
    } else if (operation->kind == Operation::Kind::Phase) {
      const auto* op = static_cast<const PhaseOperation*>(operation.get());
      instruction.opcode = "phase";
      instruction.phase = op->phase;
      current_phase = op->phase;
    } else if (operation->kind == Operation::Kind::Barrier) {
      const auto* op = static_cast<const BarrierOperation*>(operation.get());
      instruction.opcode = "barrier";
      instruction.barrierScope = op->scope;
    } else if (operation->kind == Operation::Kind::MapSurface) {
      const auto* op = static_cast<const MapSurfaceOperation*>(operation.get());
      switch (op->surfaceKind) {
        case MapSurfaceOperation::SurfaceKind::Load:
          instruction.opcode = "map_load";
          break;
        case MapSurfaceOperation::SurfaceKind::Store:
          instruction.opcode = "map_store";
          break;
        case MapSurfaceOperation::SurfaceKind::AtomicAdd:
          instruction.opcode = "map_atomic_add";
          break;
        case MapSurfaceOperation::SurfaceKind::AtomicSub:
          instruction.opcode = "map_atomic_sub";
          break;
        case MapSurfaceOperation::SurfaceKind::AtomicXor:
          instruction.opcode = "map_atomic_xor";
          break;
      }
      instruction.rid = op->rid;
      instruction.lane = op->lane;
      instruction.offset = parse_numeric_suffix(op->target);
      instruction.target = op->target;
      track_source_dependency(&instruction, op->rid);
    } else {
      const auto* op = static_cast<const MacroExpansionOperation*>(operation.get());
      instruction.opcode = "expand." + op->macroName;
    }

    instruction.phase = current_phase;
    if (last_control_index.has_value()) {
      instruction.dependencies.push_back(
          RopDependency{*last_control_index, "control"});
    }

    stream.push_back(std::move(instruction));
    const std::size_t index = stream.size() - 1;
    const RopInstruction& inserted = stream.back();

    if (!inserted.rid.empty() &&
        (inserted.opcode == "ld" || inserted.opcode == "map_load" ||
         inserted.opcode == "add" || inserted.opcode == "sub" ||
         inserted.opcode == "xor" || inserted.opcode == "mul" ||
         inserted.opcode == "and" || inserted.opcode == "or")) {
      last_writer[inserted.rid] = index;
    }
    if (inserted.opcode == "phase" || inserted.opcode == "barrier") {
      last_control_index = index;
    }
  }

  return stream;
}

BlockNode rop_stream_to_bcir_graph(const std::vector<RopInstruction>& stream) {
  BlockNode block;
  for (const RopInstruction& instruction : stream) {
    if (instruction.opcode == "ld" || instruction.opcode == "st") {
      auto op = std::make_unique<LdStOperation>(instruction.opcode == "ld");
      op->rid = instruction.rid;
      op->registryType = instruction.laneType;
      op->lane = instruction.lane;
      op->target = instruction.target;
      block.operations.push_back(std::move(op));
      continue;
    }

    if (instruction.opcode == "phase") {
      auto op = std::make_unique<PhaseOperation>();
      op->phase = instruction.phase.value_or(0);
      block.operations.push_back(std::move(op));
      continue;
    }

    if (instruction.opcode == "barrier") {
      auto op = std::make_unique<BarrierOperation>();
      op->scope = instruction.barrierScope;
      block.operations.push_back(std::move(op));
      continue;
    }

    if (instruction.opcode == "map_load" || instruction.opcode == "map_store" ||
        instruction.opcode == "map_atomic_add" ||
        instruction.opcode == "map_atomic_sub" ||
        instruction.opcode == "map_atomic_xor") {
      auto op = std::make_unique<MapSurfaceOperation>();
      if (instruction.opcode == "map_load") {
        op->surfaceKind = MapSurfaceOperation::SurfaceKind::Load;
      } else if (instruction.opcode == "map_store") {
        op->surfaceKind = MapSurfaceOperation::SurfaceKind::Store;
      } else if (instruction.opcode == "map_atomic_add") {
        op->surfaceKind = MapSurfaceOperation::SurfaceKind::AtomicAdd;
      } else if (instruction.opcode == "map_atomic_sub") {
        op->surfaceKind = MapSurfaceOperation::SurfaceKind::AtomicSub;
      } else {
        op->surfaceKind = MapSurfaceOperation::SurfaceKind::AtomicXor;
      }
      op->rid = instruction.rid;
      op->lane = instruction.lane;
      op->target = instruction.target;
      block.operations.push_back(std::move(op));
      continue;
    }

    if (instruction.opcode.rfind("lane.", 0) == 0) {
      auto op = std::make_unique<LaneOpOperation>();
      op->opcode = instruction.opcode.substr(5);
      op->rid = instruction.rid;
      op->lane = instruction.lane;
      block.operations.push_back(std::move(op));
    } else {
      auto op = std::make_unique<BinaryOpOperation>();
      op->opcode = instruction.opcode;
      op->dst = instruction.rid;
      op->dstType = instruction.laneType;
      op->lhs = instruction.lhs.empty() ? instruction.rid : instruction.lhs;
      op->rhs = instruction.rhs.empty() ? instruction.rid : instruction.rhs;
      block.operations.push_back(std::move(op));
    }
  }

  return block;
}

}  // namespace bcir
