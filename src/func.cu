#include "func.hpp"

__global__ void vadd(float* a, float* b, float* dest)
{
  dest[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

void cu_add(float* a, float* b, float* dest, size_t size)
{
  vadd<<<1, size>>>(a, b, dest);
}
