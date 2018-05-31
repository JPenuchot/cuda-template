#include "func.hpp"

__global__ void ke_add(float* a, float* b, float* dest)
{
  dest[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

__host__ void cu_add(float* a, float* b, float* dest, size_t size)
{
  ke_add <<<1, size>>> (a, b, dest);
}
