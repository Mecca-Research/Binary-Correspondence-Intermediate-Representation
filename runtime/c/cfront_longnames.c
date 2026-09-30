/* CF-BUF: every identifier below that is not an `ln_` entry is 63 characters long -- the longest either cfront
 * rail accepts, C11 5.2.4.1's 63 significant initial characters of an internal identifier -- and so is the
 * floating constant in `ln_float`. Each sits where the claim graph keeps a name: a callee after the prefix of
 * `c.call:` and `c.call.void:`, a prototyped function, a parameter, two same-named locals in disjoint blocks (the
 * emit names the second `<name>_2`), a static local, a file-scope table, a struct tag in the type of a local, a
 * pointer and a cast, the struct's members, a function-pointer member (`c.call.imember:`), a typedef, an enum
 * constant, a label (`c.goto:`, `c.label:`) and the constant (`c.fconst:`). The C twin kept 31 characters of each:
 * its digest differed from the oracle's, and its emit called a name nothing defines -- or another function. A
 * 64th character is refused on both rails; a variadic callee of 63 characters is a unit of its own, since the
 * corpus harness does not include <stdarg.h> (bcir/tests/test_c_cfront.py). */
#include <stdint.h>

typedef uint32_t unsigned_word_type_named_by_a_typedef_with_the_longest_name_ok_;
enum { enumeration_constant_named_with_every_character_c_guarantees_77 = 7 };
struct register_block_descriptor_with_a_tag_as_long_as_c_allows_here_1 {
  uint32_t first_member_of_the_register_block_descriptor_at_offset_zero_ab;
  uint16_t second_member_of_the_register_block_descriptor_after_the_first_;
};
struct operations_table_holding_one_function_pointer_member_to_call___ {
  uint32_t (*function_pointer_member_called_through_the_operations_table____)(uint32_t);
};
static uint32_t file_scope_lookup_table_of_four_words_with_a_sixty_three_chars_[4] = {3u, 5u, 7u, 11u};

static uint32_t static_callee_whose_name_fills_all_sixty_three_characters_here_(
    uint32_t parameter_of_the_callee_with_a_name_of_sixty_three_characters__) {
  return parameter_of_the_callee_with_a_name_of_sixty_three_characters__ * 3u +
         enumeration_constant_named_with_every_character_c_guarantees_77;
}

static void void_callee_that_adds_its_argument_into_the_lookup_table_entry_(uint32_t v) {
  file_scope_lookup_table_of_four_words_with_a_sixty_three_chars_[v & 3u] += v;
}

uint32_t prototyped_function_declared_before_its_definition_as_c_allows_(uint32_t v);
uint32_t prototyped_function_declared_before_its_definition_as_c_allows_(uint32_t v) { return v ^ 0x5Au; }

uint32_t ln_calls(uint32_t s) {
  void_callee_that_adds_its_argument_into_the_lookup_table_entry_(s);
  return static_callee_whose_name_fills_all_sixty_three_characters_here_(s) +
         prototyped_function_declared_before_its_definition_as_c_allows_(s) +
         file_scope_lookup_table_of_four_words_with_a_sixty_three_chars_[s & 3u];
}

uint32_t ln_struct(uint32_t s) {
  struct register_block_descriptor_with_a_tag_as_long_as_c_allows_here_1 d;
  d.first_member_of_the_register_block_descriptor_at_offset_zero_ab = s * 5u;
  d.second_member_of_the_register_block_descriptor_after_the_first_ = (uint16_t)(s + 9u);
  const void *raw = &d;
  const struct register_block_descriptor_with_a_tag_as_long_as_c_allows_here_1
      *pointer_to_the_register_block_descriptor_cast_from_void_pointer =
          (const struct register_block_descriptor_with_a_tag_as_long_as_c_allows_here_1 *)raw;
  return pointer_to_the_register_block_descriptor_cast_from_void_pointer
             ->first_member_of_the_register_block_descriptor_at_offset_zero_ab +
         pointer_to_the_register_block_descriptor_cast_from_void_pointer
             ->second_member_of_the_register_block_descriptor_after_the_first_;
}

uint32_t ln_typedef(uint32_t s) {
  unsigned_word_type_named_by_a_typedef_with_the_longest_name_ok_ w =
      s + enumeration_constant_named_with_every_character_c_guarantees_77;
  return w * 2u;
}

uint32_t ln_shadow(uint32_t s) {
  uint32_t r = 1u;
  {
    uint32_t local_variable_declared_twice_in_two_disjoint_blocks_of_scope__ = s + 1u;
    r += local_variable_declared_twice_in_two_disjoint_blocks_of_scope__;
  }
  {
    uint32_t local_variable_declared_twice_in_two_disjoint_blocks_of_scope__ = s * 2u;
    r ^= local_variable_declared_twice_in_two_disjoint_blocks_of_scope__;
  }
  return r;
}

uint32_t ln_static(uint32_t s) {
  static uint32_t static_local_counter_that_keeps_its_value_between_every_call___ = 5u;
  static_local_counter_that_keeps_its_value_between_every_call___ += s;
  return static_local_counter_that_keeps_its_value_between_every_call___;
}

uint32_t ln_goto(uint32_t s) {
  if (s > 10u) goto label_at_the_end_of_the_function_that_the_goto_jumps_forward_to;
  s += 3u;
label_at_the_end_of_the_function_that_the_goto_jumps_forward_to:
  return s * 2u;
}

double ln_float(double x) { return x * 1.00000000000000000000000000000000000000000000000000000000e-300; }

uint32_t ln_imember(uint32_t s) {
  struct operations_table_holding_one_function_pointer_member_to_call___ o;
  o.function_pointer_member_called_through_the_operations_table____ =
      static_callee_whose_name_fills_all_sixty_three_characters_here_;
  return o.function_pointer_member_called_through_the_operations_table____(s);
}

uint32_t ln_entry(uint32_t s) {   /* the entry: pure, so the corpus harness can run it on seeded inputs */
  return static_callee_whose_name_fills_all_sixty_three_characters_here_(s) + ln_struct(s) + ln_typedef(s) +
         ln_shadow(s) + ln_goto(s) + ln_imember(s) +
         file_scope_lookup_table_of_four_words_with_a_sixty_three_chars_[s & 3u];
}
