#include <iostream>
#include <numeric>
#include <vector>
#include <thread>

#include <cuda.h>

#include "funcs.hpp"

using namespace std;

int main()
{
  vector<float> va(100);
  vector<float> vb(100);

  thread fill_va([&]() { fill(va.begin(), va.end(), 0.f); });
  thread fill_vb([&]() { iota(vb.begin(), vb.end(), 0.f); });

  float* dev_a = nullptr;
  float* dev_b = nullptr;

  fill_va.join();
  fill_vb.join();

  cudaMalloc(&dev_a, va.size() * sizeof(float));
  cudaMalloc(&dev_b, vb.size() * sizeof(float));

  cudaMemcpy( dev_a, va.data()
            , va.size() * sizeof(float)
            , cudaMemcpyHostToDevice);

  cudaMemcpy( dev_b, vb.data()
            , vb.size() * sizeof(float)
            , cudaMemcpyHostToDevice);


  cu_add(dev_a, dev_b, dev_a, va.size());

  cudaMemcpy( dev_a, va.data()
            , va.size() * sizeof(float), cudaMemcpyDeviceToHost);

  for(auto& e : va) cout << e << '\n';

  return 0;
}
