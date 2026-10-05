/* rid->extent tamper-evidence (#extentassert, docs §5.12): BCIR_EXTENT_ASSERT(arr, n) is
 * _Static_assert(n == sizeof(arr)/sizeof(arr[0])), so a guard's extent is tied to its array's storage
 * at compile time, with no runtime cost and no registry. Built as is, the extent is right and the unit
 * compiles; tools/c/sections/extentassert.sh compiles it again with -DBCIR_TEST_EXTENT=99u, a tampered
 * extent, which must fail to compile on the assertion itself (BUILD-2j, docs/BCIR_BUILD_ROADMAP.md). */
#include "bcir_quarantine.h"
#ifndef BCIR_TEST_EXTENT
#define BCIR_TEST_EXTENT 8u
#endif
static unsigned a[8];
int main(void){ BCIR_EXTENT_ASSERT(a, BCIR_TEST_EXTENT); return (int)a[0]; }
