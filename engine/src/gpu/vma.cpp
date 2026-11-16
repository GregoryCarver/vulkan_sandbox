// Keep the allocator implementation in one translation unit. Its dispatch is
// obtained from the same loader used by Device on Windows and Android.
#define VMA_IMPLEMENTATION
#include "vma_config.hpp"
