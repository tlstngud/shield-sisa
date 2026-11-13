// avx512 build of the SHIELD-V CPU kernels (see isa_impl.hpp). Compiled with ISA flags for this file only;
// only called after a runtime CPU feature check.
#define SHV_LEVEL 3
#include "isa_impl.hpp"

extern "C" const shv::IsaOps* shv_isa_ops_avx512(void) { return &ops; }
