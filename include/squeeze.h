#ifndef SQUEEZE_H
#define SQUEEZE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int8_t *quantized_data; /* Buffer continuo de n * dim bytes */
    float  *scales;         /* Factor de escala por vector (tamaño n) */
    size_t n;
    size_t dim;
} QuantizedDataset;

/*
    Compresses n vectors of dim dimensions from fp32 to int8 components
*/
QuantizedDataset* quantize_fp32_to_int8(const float *data, size_t n, size_t dim);

void compute_quantized_dot_product_cuda(
    const int8_t *d_query, float query_scale,
    const QuantizedDataset *d_dataset,
    float *d_out_distances
);

/* Liberación */
void free_quantized_dataset(QuantizedDataset *ds);

#ifdef __cplusplus
}
#endif

#endif /* SQUEEZE_H */
