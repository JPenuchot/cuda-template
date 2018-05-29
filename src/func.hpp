#pragma once

__global__ void vadd(float* a, float* b, float* dest);
void cu_add(float* a, float* b, float* dest, size_t size);
