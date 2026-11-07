#include "../include/squeeze.h"
#include <cuda_runtime.h>

__global__ void computeQuantizedDotProductKernel(const int8_t *d_query, float query_scale, const QuantizedDataset *d_dataset, float *d_out_distances)
{

}

extern "C" void compute_quantized_dot_product_cuda(const int8_t *d_query, float query_scale, const QuantizedDataset *d_dataset, float *d_out_distances)
{

}