#include "numER/array.h"
#include "numER/iterator.h"
#include "numER/broadcasting.h"
#include <stdlib.h>

int nc_compute_broadcast_shape(int n_arrays, NDArray** arrays, int* out_ndim, size_t** out_shape)
{
    int i, axis;
    int max_ndim = 0;

    if (!arrays || !out_ndim || !out_shape || n_arrays <= 0) return 0;

    for (i = 0; i < n_arrays; ++i) {
        if (!arrays[i]) return 0;
        if (arrays[i]->ndim < 0) return 0;
        if (arrays[i]->ndim > 0 && !arrays[i]->shape) return 0;
        if (arrays[i]->ndim > max_ndim) max_ndim = arrays[i]->ndim;
    }

    size_t* result_shape = (size_t*)malloc((size_t)(max_ndim > 0 ? max_ndim : 1) * sizeof(size_t));
    if (!result_shape) return 0;

    for (axis = 0; axis < max_ndim; ++axis) result_shape[axis] = 1;

    for (i = 0; i < n_arrays; ++i) {
        NDArray* arr = arrays[i];
        for (axis = 0; axis < max_ndim; ++axis) {
            const int arr_axis = axis - (max_ndim - arr->ndim);
            const size_t arr_dim = (arr_axis >= 0) ? arr->shape[arr_axis] : 1;
            const size_t out_dim = result_shape[axis];

            if (out_dim == 1) {
                result_shape[axis] = arr_dim;
            } else if (arr_dim != 1 && arr_dim != out_dim) {
                free(result_shape);
                return 0;
            }
        }
    }

    *out_ndim = max_ndim;
    *out_shape = (max_ndim > 0) ? result_shape : NULL;
    if (max_ndim == 0) free(result_shape);
    return 1;
}

int nc_broadcast_check(int n_arrays, NDArray** arrays)
{
    int out_ndim;
    size_t* out_shape;

    if (!nc_compute_broadcast_shape(n_arrays, arrays, &out_ndim, &out_shape)) return 0;
    free(out_shape);
    return 1;
}

int* nc_compute_broadcast_strides(const NDArray* arr, int target_ndim, const size_t* target_shape)
{
    int axis;
    int* result_strides;

    if (!arr || target_ndim < 0) return NULL;
    if (target_ndim > 0 && !target_shape) return NULL;
    if (arr->ndim < 0 || arr->ndim > target_ndim) return NULL;
    if (arr->ndim > 0 && (!arr->shape || !arr->strides)) return NULL;

    result_strides = (int*)malloc((size_t)(target_ndim > 0 ? target_ndim : 1) * sizeof(int));
    if (!result_strides) return NULL;

    for (axis = 0; axis < target_ndim; ++axis) {
        const int arr_axis = axis - (target_ndim - arr->ndim);
        if (arr_axis < 0) {
            result_strides[axis] = 0;
            continue;
        }

        const size_t arr_dim = arr->shape[arr_axis];
        const size_t tgt_dim = target_shape[axis];
        if (arr_dim == tgt_dim) {
            result_strides[axis] = arr->strides[arr_axis];
        } else if (arr_dim == 1) {
            result_strides[axis] = 0;
        } else {
            free(result_strides);
            return NULL;
        }
    }

    return result_strides;
}
