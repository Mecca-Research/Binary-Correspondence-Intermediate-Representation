; Packed structs remove ABI padding and may force smaller alignment.
source_filename = "packed-structs.c"

%Packed = type <{ i8, i32 }>

@packed_global = global %Packed <{ i8 1, i32 287454020 }>, align 1

define i32 @load_packed_field() {
entry:
  %field = getelementptr inbounds %Packed, ptr @packed_global, i32 0, i32 1
  %value = load i32, ptr %field, align 1
  ret i32 %value
}
