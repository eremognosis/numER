#include "numER/iterator.h"
#include "numER/array.h"
#include <stdlib.h>

static void nc_iter_cleanup_partial(NCIterator* iter, int allocated_arrays)
{
    int i;
    if (!iter) return;

    if (iter->strides) {
        for (i = 0; i < allocated_arrays; ++i) {
            free(iter->strides[i]);
        }
    }

    if (iter->backstrides) {
        for (i = 0; i < allocated_arrays; ++i) {
            free(iter->backstrides[i]);
        }
    }

    free(iter->shape);
    free(iter->coordinates);
    free(iter->data_ptrs);
    free(iter->strides);
    free(iter->backstrides);
    free(iter);
}



NCIterator* nc_iter_new(int n_arrays, NDArray** arrays)
{
    int i, axis;
    int max_ndim = 0;
    NCIterator* iter;

    if (!arrays || n_arrays <= 0) return NULL;
    for (i = 0; i < n_arrays; ++i) {
        if (!arrays[i]) return NULL;

        /// to set the maximm n_dim
        if (arrays[i]->ndim > max_ndim) max_ndim = arrays[i]->ndim;
    }

    iter = (NCIterator*)calloc(1, sizeof(NCIterator));
    if (!iter) return NULL; /// mem alloc failed

    iter->ndim = max_ndim;
    iter->n_arrays = n_arrays;
    iter->shape = (size_t*)malloc((size_t)max_ndim * sizeof(size_t));
    iter->coordinates = (size_t*)calloc((size_t)max_ndim, sizeof(size_t));
    iter->data_ptrs = (uint8_t**)malloc((size_t)n_arrays * sizeof(uint8_t*));
    iter->strides = (int**)calloc((size_t)n_arrays, sizeof(int*));
    iter->backstrides = (int**)calloc((size_t)n_arrays, sizeof(int*));

    if ((max_ndim > 0) && (!iter->shape || !iter->coordinates)) {
        nc_iter_cleanup_partial(iter, 0);
        return NULL;
    }
    if (!iter->data_ptrs || !iter->strides || !iter->backstrides) {
        nc_iter_cleanup_partial(iter, 0);
        return NULL;
    }

    for (axis = 0; axis < max_ndim; ++axis) iter->shape[axis] = 1;

    for (i = 0; i < n_arrays; ++i) {
        NDArray* arr = arrays[i];
        for (axis = 0; axis < max_ndim; ++axis) {
            int arr_axis = axis - (max_ndim - arr->ndim);
            size_t arr_dim = (arr_axis >= 0) ? arr->shape[arr_axis] : 1;
            size_t out_dim = iter->shape[axis];

            if (out_dim == 1) {
                iter->shape[axis] = arr_dim;
            } else if (arr_dim != 1 && arr_dim != out_dim) {
                nc_iter_cleanup_partial(iter, i);
                return NULL;
            }
        }
    }

    iter->size = 1;
    for (axis = 0; axis < max_ndim; ++axis) {
        iter->size *= iter->shape[axis];
    }
    iter->index = 0;

    for (i = 0; i < n_arrays; ++i) {
        NDArray* arr = arrays[i];
        iter->data_ptrs[i] = (uint8_t*)arr->data + arr->offset;
        iter->strides[i] = (int*)malloc((size_t)max_ndim * sizeof(int));
        iter->backstrides[i] = (int*)malloc((size_t)max_ndim * sizeof(int));
        if ((max_ndim > 0) && (!iter->strides[i] || !iter->backstrides[i])) {
            nc_iter_cleanup_partial(iter, i + 1);
            return NULL;
        }

        for (axis = 0; axis < max_ndim; ++axis) {
            int arr_axis = axis - (max_ndim - arr->ndim);
            size_t arr_dim = (arr_axis >= 0) ? arr->shape[arr_axis] : 1;
            int stride = 0;
            if (arr_axis >= 0 && !(arr_dim == 1 && iter->shape[axis] != 1)) {
                stride = arr->strides[arr_axis];
            }
            iter->strides[i][axis] = stride;
            iter->backstrides[i][axis] = stride * (int)(iter->shape[axis] ? (iter->shape[axis] - 1) : 0);
        }
    }

    return iter;
}


void nc_iter_free(NCIterator *iter)
{
    if (!iter) return;

    free(iter->shape);
    free(iter->coordinates);
    free(iter->data_ptrs);
    free(iter->strides);
    free(iter->backstrides);
    free(iter);

}



int nc_iter_next(NCIterator *iter)
{
    if (!iter) return -1;
    iter->index++;

    int axis, i;

    for (axis=iter->ndim-1; axis >=0; --axis)
    {
        iter->coordinates[axis]++;


        /// Case 1:  we arent at the end yet
        if (iter->coordinates[axis] < iter->shape[axis])
        {
            for (i=0; i<iter->n_arrays;i++)iter->data_ptrs[i]+=iter->strides[i][axis];
            break; //done
        }

        else
        {
            iter->coordinates[axis]=0;
            for (i = 0; i < iter->n_arrays; ++i) {
                iter->data_ptrs[i] -= iter->backstrides[i][axis];
            }
        }
    }

    return 1;
}



void* nc_iter_get_ptr(NCIterator *iter, int array_index)
{
    if (!iter) return NULL;
    if (array_index < 0 || array_index >= iter->n_arrays) return NULL;
    if (!iter->data_ptrs) return NULL;
    return (void*)iter->data_ptrs[array_index];
}
