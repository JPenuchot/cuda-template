#include <iostream>
#include <numeric>
#include <vector>

#include <cuda.h>

using namespace std;

template<typename T>
__global__ void vadd(T* a, T* b, T* dest)
{
  dest[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

template<typename T>
void cu_add(T* a, T* b, T* dest, size_t size)
{
  vadd<<<1, size>>>(a, b, dest);
}

int main()
{
  vector<float> va(100);
  vector<float> vb(100);

  fill(va.begin(), va.end(), 0.f);
  iota(vb.begin(), vb.end(), 0.f);

  float* dev_a = nullptr;
  float* dev_b = nullptr;

  cudaMalloc(&dev_a, va.size() * sizeof(float));
  cudaMalloc(&dev_b, vb.size() * sizeof(float));

  cudaMemcpy( dev_a, va.data()
            , va.size() * sizeof(float)
            , cudaMemcpyHostToDevice);

  cudaMemcpy( dev_b, vb.data()
            , vb.size() * sizeof(float)
            , cudaMemcpyHostToDevice);


  cu_add(dev_a, dev_b, dev_a, va.size());

  cudaMemcpy( va.data(), dev_a
            , va.size() * sizeof(float), cudaMemcpyDeviceToHost);

  for(auto& e : va) cout << e << '\n';

  return 0;
}
