; null, undef, poison, and freeze are distinct constant/value forms.
source_filename = "null-undef-poison-freeze.c"

@maybe_ptr = global ptr null, align 8
@undef_seed = global i32 undef, align 4
@poison_seed = global i32 poison, align 4

define i32 @freeze_poison_constant() {
entry:
  %stable = freeze i32 poison
  ret i32 %stable
}
