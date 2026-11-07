#include "../include/squeeze.h"
#include <stdlib.h>
#include <math.h>

QuantizedDataset* quantize_fp32_to_int8(const float *data, size_t n, size_t dim)
{
    if (data == NULL || n <= 0 || dim <= 0) return NULL;
    
    QuantizedDataset *out = malloc(sizeof *out);
    if (out == NULL) return NULL;

    if ((out->quantized_data = malloc(n * dim * sizeof *out->quantized_data)) == NULL)
    {
        free(out);
        return NULL;
    }
    if ((out->scales = malloc(n * sizeof *out->scales)) == NULL)
    {
        free(out->quantized_data);
        free(out->scales);
        free(out);
        return NULL;
    }
    out->n = n;
    out->dim = dim;

    for (size_t i = 0; i < n; i++)
    {
        float max_abs = 0;
        for (size_t j = 0; j < dim; j++)
        {
            float abs = fabs(data[i * dim + j]);
            if (abs > max_abs) max_abs = abs;
        }
        float scale = max_abs / 127;

        out->scales[i] = scale;

        if (scale == 0)
        {
            for (size_t j = 0; j < dim; j++)
            {
                out->quantized_data[i * dim + j] = 0;
            }
        }
        else {
            for (size_t j = 0; j < dim; j++)
            {
                float elem = data[i * dim + j];
                double q = round(elem / scale);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                out->quantized_data[i * dim + j] = (int8_t)q;
            }
        }
    }

    return out;
}

void free_quantized_dataset(QuantizedDataset *ds)
{
    if (ds == NULL) return;
    
    free(ds->quantized_data);
    free(ds->scales);
    free(ds);
}