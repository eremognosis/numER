#include "numER/array.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>


size_t nc_dtype_size(DataType dtype)
{
    switch (dtype)
    {
    case NC_FLOAT:
        return sizeof(float);
        break;
    case NC_INT:
        return sizeof(int);
        break;

    case NC_DOUBLE:
        return sizeof(double);
        break;

    case NC_LONGINT:
        return sizeof(long int);

        break;
    case NC_COMPLEX:
        return sizeof(double complex);
        break;

    default:
        return 0;
    }
}

NDArray* nc_array_empty(int ndim, const size_t* shape, DataType dtype)
{
    int i;
    size_t total_size = 1;

    if (ndim < 0) return NULL;
    if (ndim > 0 && !shape) return NULL;

    size_t dtype_size = nc_dtype_size(dtype);
    if (dtype_size == 0) return NULL;

    NDArray* array = (NDArray*)calloc(1, sizeof(NDArray));
    if (!array) return NULL;

    array->ndim = ndim;
    array->dtype = dtype;
    array->offset = 0;

    if (ndim > 0)
    {
        array->shape = (size_t*)malloc((size_t)ndim * sizeof(size_t));
        array->strides = (int*)malloc((size_t)ndim * sizeof(int));
        if (!array->shape || !array->strides)
        {
            free(array->shape);
            free(array->strides);
            free(array);
            return NULL;
        }

        for (i = 0; i < ndim; ++i)
        {
            array->shape[i] = shape[i];
            total_size *= shape[i];
        }

        if (ndim > 0)
        {
            size_t stride = dtype_size;
            for (i = ndim - 1; i >= 0; --i)
            {
                if (stride > (size_t)INT_MAX)
                {
                    free(array->shape);
                    free(array->strides);
                    free(array);
                    return NULL;
                }
                array->strides[i] = (int)stride;
                stride *= array->shape[i];
                if (i == 0) break;
            }
        }
    }

    array->size = total_size;
    if (total_size == 0)
    {
        array->data = NULL;
        return array;
    }

    array->data = malloc(total_size * dtype_size);
    if (!array->data)
    {
        free(array->shape);
        free(array->strides);
        free(array);
        return NULL;
    }

    return array;
}

NDArray* nc_array_zeros(int ndim, const size_t* shape, DataType dtype)
{
    NDArray* array = nc_array_empty(ndim, shape, dtype);
    if (!array) return NULL;

    if (array->data && array->size > 0)
    {
        memset(array->data, 0, array->size * nc_dtype_size(dtype));
    }

    return array;
}

void nc_array_free(NDArray* array)
{
    if (!array) return;
    free(array->data);
    free(array->shape);
    free(array->strides);
    free(array);
}

NDArray* nc_array_view(const NDArray* source)
{
    if (!source) return NULL;
    if (!source->data) return NULL;
    NDArray* array = nc_array_empty(source->ndim, source->shape, source->dtype);
    if (!array) return NULL;

    array->data = source->data;
    return array;
}

NDArray* nc_array_reshape(const NDArray* source, int new_ndim, const size_t* new_shape)
{
    int i;
    size_t new_size = 1;

    if (!source) return NULL;
    if (new_ndim < 0) return NULL;
    if (new_ndim > 0 && !new_shape) return NULL;
    if (!source->data && source->size > 0) return NULL;
    if (!nc_array_is_contiguous(source)) return NULL;

    for (i = 0; i < new_ndim; ++i)
    {
        new_size *= new_shape[i];
    }
    if (new_size != source->size) return NULL; // bruh dont make up data you aint polittician

    size_t dtype_size = nc_dtype_size(source->dtype);
    if (dtype_size == 0) return NULL;

    NDArray* reshaped = (NDArray*)calloc(1, sizeof(NDArray));
    if (!reshaped) return NULL;

    reshaped->ndim = new_ndim;
    reshaped->dtype = source->dtype;
    reshaped->size = source->size;
    reshaped->offset = source->offset;
    reshaped->data = source->data;

    if (new_ndim > 0)
    {
        size_t stride = dtype_size;
        reshaped->shape = (size_t*)malloc((size_t)new_ndim * sizeof(size_t));
        reshaped->strides = (int*)malloc((size_t)new_ndim * sizeof(int));
        if (!reshaped->shape || !reshaped->strides)
        {
            free(reshaped->shape);
            free(reshaped->strides);
            free(reshaped);
            return NULL;
        }

        for (i = 0; i < new_ndim; ++i)
        {
            reshaped->shape[i] = new_shape[i];
        }

        for (i = new_ndim - 1; i >= 0; --i)
        {
            if (stride > (size_t)INT_MAX)
            {
                free(reshaped->shape);
                free(reshaped->strides);
                free(reshaped);
                return NULL;
            }
            reshaped->strides[i] = (int)stride;
            stride *= reshaped->shape[i];
            if (i == 0) break;
        }
    }

    return reshaped;
}

bool nc_array_is_contiguous(const NDArray* array)
{
    if (!array) return false;
    if (array->ndim < 0) return false;

    size_t dtype_size = nc_dtype_size(array->dtype);
    if (dtype_size == 0) return false;

    if (array->ndim == 0) return true;
    if (!array->shape || !array->strides) return false;

    size_t expected_stride = dtype_size;
    for (int i = array->ndim - 1; i >= 0; --i)
    {
        if (array->shape[i] == 0) return true;
        if (array->strides[i] != (int)expected_stride) return false;
        expected_stride *= array->shape[i];
        if (i == 0) break;
    }

    return true;
}

static const char* nc_dtype_name(DataType dtype)
{
    switch (dtype)
    {
    case NC_FLOAT:
        return "float";
    case NC_INT:
        return "int";
    case NC_COMPLEX:
        return "complex";
    case NC_DOUBLE:
        return "double";
    case NC_LONGINT:
        return "longint";
    default:
        return "unknown";
    }
}

void nc_array_print(const NDArray* array)
{
    if (!array)
    {
        printf("NDArray(NULL)\n");
        return;
    }

    printf("NDArray(shape=(");
    for (int i = 0; i < array->ndim; ++i)
    {
        printf("%zu", array->shape[i]);
        if (i + 1 < array->ndim) printf(", ");
    }
    printf("), strides=(");
    for (int i = 0; i < array->ndim; ++i)
    {
        printf("%d", array->strides[i]);
        if (i + 1 < array->ndim) printf(", ");
    }
    printf("), dtype=%s, size=%zu)\n", nc_dtype_name(array->dtype), array->size);

    if (!array->data || array->size == 0)
    {
        printf("[]\n");
        return;
    }

    const uint8_t* base = (const uint8_t*)array->data + array->offset;
    printf("[");

    for (size_t flat_idx = 0; flat_idx < array->size; ++flat_idx)
    {
        ptrdiff_t byte_offset = 0;
        size_t rem = flat_idx;

        for (int axis = array->ndim - 1; axis >= 0; --axis)
        {
            const size_t coord = rem % array->shape[axis];
            rem /= array->shape[axis];
            byte_offset += (ptrdiff_t)coord * array->strides[axis];
            if (axis == 0) break;
        }

        const uint8_t* elem_ptr = base + byte_offset;

        switch (array->dtype)
        {
        case NC_FLOAT:
            printf("%g", (double)(*(const float*)elem_ptr));
            break;
        case NC_INT:
            printf("%d", *(const int*)elem_ptr);
            break;
        case NC_DOUBLE:
            printf("%g", *(const double*)elem_ptr);
            break;
        case NC_LONGINT:
            printf("%ld", *(const long int*)elem_ptr);
            break;
        case NC_COMPLEX:
            {
                const double complex v = *(const double complex*)elem_ptr;
                printf("(%g%+gi)", creal(v), cimag(v));
                break;
            }
        default:
            printf("<?>");
            break;
        }

        if (flat_idx + 1 < array->size) printf(", ");
    }

    printf("]\n");
}
