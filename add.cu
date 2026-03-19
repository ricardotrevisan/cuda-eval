#include <iostream>
#include <math.h>

// Kernel
__global__
void add(int n, float *x, float *y)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Unified Memory
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // Inicializa na CPU
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Prefetch para GPU (ESSENCIAL)
  int device;
  cudaGetDevice(&device);

  cudaMemPrefetchAsync(x, N*sizeof(float), device);
  cudaMemPrefetchAsync(y, N*sizeof(float), device);

  // (opcional mas bom hábito)
  cudaDeviceSynchronize();

  // Configuração de execução
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  add<<<numBlocks, blockSize>>>(N, x, y);

  cudaDeviceSynchronize();

  // Verificação
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  }

  std::cout << "Max error: " << maxError << std::endl;

  // Liberação
  cudaFree(x);
  cudaFree(y);

  return 0;
}