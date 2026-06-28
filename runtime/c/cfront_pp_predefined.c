/* L7 adversarial preprocessor corpus -- the dynamic predefined macro `__LINE__` and `__has_*`
 * feature-test operators (C 6.10.1, 6.8.1). `__LINE__` is the presumed line number at the point of
 * use; `XSTR(__LINE__)` (two-level) spells the NUMBER, not the token, exercising argument prescan of a
 * dynamic macro. `__has_attribute` for a standard attribute is reported in `#if`. Deliberately uses ONLY
 * single-valued predefineds: `__LINE__` is deterministic and `__has_attribute(deprecated)` /
 * `(__packed__)` agree across clang and gcc -- so this is a fair single-truth differential. (The
 * implementation-defined `__STDC_VERSION__` VALUE lives in cfront_pp_stdcver.c, which the fairness gate
 * excludes; it is kept out of this fixture on purpose.) */
#define STR(x) #x
#define XSTR(x) STR(x)
here_line __LINE__
spelled XSTR(__LINE__)
#if __has_attribute(packed)
has_packed
#endif
#if __has_attribute(__packed__)
has_packed_underscored
#endif
#if !__has_attribute(this_is_not_a_real_attribute_xyz)
no_bogus_attribute
#endif
#if defined(__has_attribute)
has_attribute_probe_works
#endif
