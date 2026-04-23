#include "bcir/lowering.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <regex>
#include <set>
#include <string>
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

}  // namespace bcir
