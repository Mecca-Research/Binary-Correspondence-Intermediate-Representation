source_filename = "bcir_rehydrate.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

define i32 @bcir.rehydrate.decide(i64 %old_topo,
                                  i64 %new_topo,
                                  i64 %old_map,
                                  i64 %new_map,
                                  i64 %old_data,
                                  i64 %new_data,
                                  i64 %thermal_delta) {
entry:
  %topo_changed = icmp ne i64 %old_topo, %new_topo
  br i1 %topo_changed, label %replan, label %check_map
check_map:
  %map_changed = icmp ne i64 %old_map, %new_map
  br i1 %map_changed, label %repack, label %check_data
check_data:
  %data_changed = icmp ne i64 %old_data, %new_data
  br i1 %data_changed, label %patch, label %check_thermal
check_thermal:
  %hot = icmp ugt i64 %thermal_delta, 10
  br i1 %hot, label %repack, label %keep
patch:
  ret i32 1
repack:
  ret i32 2
replan:
  ret i32 3
keep:
  ret i32 0
}
