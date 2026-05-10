#include "numER/ops.h"
#include "numER/iterator.h"
#include <math.h>
#include <complex.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>


#define NC_EQ_FLOAT_ATOL 1e-8f
#define NC_EQ_FLOAT_RTOL 1e-5f
#define NC_EQ_DOUBLE_ATOL 1e-12
#define NC_EQ_DOUBLE_RTOL 1e-8
static int nc_isclose_float(float lhs, float rhs)
{
    const float abs_lhs = fabsf(lhs);
    const float abs_rhs = fabsf(rhs);
    const float rel_ref = fmaxf(abs_lhs, abs_rhs);
    const float tol = NC_EQ_FLOAT_ATOL + NC_EQ_FLOAT_RTOL * abs_rhs + FLT_EPSILON * rel_ref;
    return fabsf(lhs - rhs) <= tol;
}

static int nc_isclose_double(double lhs, double rhs)
{
    const double abs_lhs = fabs(lhs);
    const double abs_rhs = fabs(rhs);
    const double rel_ref = fmax(abs_lhs, abs_rhs);
    const double tol = NC_EQ_DOUBLE_ATOL + NC_EQ_DOUBLE_RTOL * abs_rhs + DBL_EPSILON * rel_ref;
    return fabs(lhs - rhs) <= tol;
}

static int nc_isclose_complex(double complex lhs, double complex rhs)
{
    const double abs_lhs = cabs(lhs);
    const double abs_rhs = cabs(rhs);
    const double rel_ref = fmax(abs_lhs, abs_rhs);
    const double tol = NC_EQ_DOUBLE_ATOL + NC_EQ_DOUBLE_RTOL * abs_rhs + DBL_EPSILON * rel_ref;
    return cabs(lhs - rhs) <= tol;
}

void nc_apply_binary(const NDArray* a, const NDArray* b, NDArray* out, nc_binary_op op)
{
    if (!a || !b || !out || !op) return;

    NDArray* arrays[3];
    arrays[0] = a;
    arrays[1] = b;
    arrays[2] = out;

    NCIterator* iter = nc_iter_new(3, arrays);
    if (!iter) return;

    for (size_t i = 0; i < iter->size; ++i)
    {
        op(nc_iter_get_ptr(iter, 2), nc_iter_get_ptr(iter, 0), nc_iter_get_ptr(iter, 1), a->dtype);
        if (i + 1 < iter->size)
        {
            nc_iter_next(iter);
        }
    }

    nc_iter_free(iter);
}

void nc_apply_unary(const NDArray* a, NDArray* out, nc_unary_op op)
{
    NDArray* arrays[2];
    if (!a || !out || !op) return;

    arrays[0] = (NDArray*)a;
    arrays[1] = out;
    NCIterator* iter = nc_iter_new(2, arrays);
    if (!iter) return;

    for (size_t i = 0; i < iter->size; ++i)
    {
        op(nc_iter_get_ptr(iter, 1), nc_iter_get_ptr(iter, 0), a->dtype);
        if (i + 1 < iter->size)
        {
            nc_iter_next(iter);
        }
    }

    nc_iter_free(iter);
}


#define DEFINE_BINARY_WORKER(name, op) \
static void name##_worker(void* out, const void* a, const void* b, DataType dtype) { \
switch (dtype) { \
case NC_INT:    *(int*)out = *(const int*)a op *(const int*)b; break; \
case NC_FLOAT:  *(float*)out = *(const float*)a op *(const float*)b; break; \
case NC_DOUBLE: *(double*)out = *(const double*)a op *(const double*)b; break; \
default: break; \
} \
}

DEFINE_BINARY_WORKER(add, +)

NDArray* nc_add(const NDArray* a, const NDArray* b, NDArray* out)
{
    nc_apply_binary(a, b, out, add_worker);
    return out;
}

DEFINE_BINARY_WORKER(sub, -)

NDArray* nc_subtract(const NDArray* a, const NDArray* b, NDArray* out)
{
    nc_apply_binary(a, b, out, sub_worker);
    return out;
}

DEFINE_BINARY_WORKER(mul, *)

NDArray* nc_multiply(const NDArray* a, const NDArray* b, NDArray* out)
{
    nc_apply_binary(a, b, out, mul_worker);
    return out;
}

DEFINE_BINARY_WORKER(div, /);

NDArray* nc_divide(const NDArray* a, const NDArray* b, NDArray* out)
{
    nc_apply_binary(a, b, out, div_worker);
    return out;
}


static void powerop(void* out, const void* a, const void* b, DataType dtype)
{
    switch (dtype)
    {
    case NC_INT:
        *(int*)out = pow(*(const int*)a, *(const int*)b);
        break;
    case NC_FLOAT:
        *(float*)out = pow(*(const float*)a, *(const float*)b);
        break;
    case NC_DOUBLE:
        *(double*)out = pow(*(const double*)a, *(const double*)b);
        break;
    case NC_LONGINT:
        *(long int*)out = pow(*(const long int*)a, *(const long int*)b);
        break;

    case NC_COMPLEX:
        *(double complex*)out = cpow(*(const double complex*)a, *(const double complex*)b);
        break;


    default:
        break;
    }
}


NDArray* nc_power(const NDArray* a, const NDArray* b, NDArray* out)
{
    nc_apply_binary(a, b, out, powerop);
    return out;
}


static void modop(void* out, const void* a, const void* b, DataType dtype)
{
    switch (dtype)
    {
    case NC_INT:
        *(int*)out = (*(const int*)a % *(const int*)b);
        break;
    case NC_FLOAT:
        *(float*)out = fmod(*(const float*)a, *(const float*)b);
        break;
    case NC_DOUBLE:
        *(double*)out = fmod(*(const double*)a, *(const double*)b);
        break;
    case NC_LONGINT:
        *(long int*)out = (*(const long int*)a % *(const long int*)b);
        break;

    case NC_COMPLEX:
        fprintf(stderr, "ATTEMPTED COMPLEX MODULO");
        break;


    default:
        break;
    }
}


NDArray* nc_remainder(const NDArray* a, const NDArray* b, NDArray* out)
{
    nc_apply_binary(a, b, out, modop);
    return out;
}


#define DEFINE_UNARY(name, op) \
static void name##_worker(void* out, const void* a, DataType dtype) { \
switch (dtype) { \
case NC_INT:    *(int*)out = op( *(const int*)a); break; \
case NC_FLOAT:  *(float*)out = op(*(const float*)a) ; break; \
case NC_DOUBLE: *(double*)out = op(*(const double*)a) ; break; \
default: break; \
} \
}


DEFINE_UNARY(neg, -);

NDArray* nc_negative(const NDArray* a, NDArray* out)
{
    nc_apply_unary(a, out, neg_worker);
    return out;
}

static void absol_worker(void* out, const void* a, DataType dtype)
{
    switch (dtype)
    {
    case NC_INT:
        *(int*)out = abs(*(const int*)a);
        break;

    case NC_FLOAT:
        *(float*)out = fabs(*(const float*)a);
        break;

    case NC_DOUBLE:
        *(double*)out = fabs(*(const double*)a);
        break;
    case NC_LONGINT:
        *(long int*)out = labs(*(const long int*)a);
        break;

    case NC_COMPLEX:
        *(double complex*)out = cabs(*(const double complex*)a);
        break;

    default: break;
    }
}

NDArray* nc_absolute(const NDArray* a, NDArray* out)
{
    nc_apply_unary(a, out, absol_worker);
    return out;
}


static void sqrt_worker(void* out, const void* a, DataType dtype)
{
    switch (dtype)
    {
    case NC_INT:
        *(int*)out = sqrt(*(const int*)a);
        break;

    case NC_FLOAT:
        *(float*)out = sqrtf(*(const float*)a);
        break;

    case NC_DOUBLE:
        *(double*)out = sqrtf(*(const double*)a);
        break;
    case NC_LONGINT:
        *(long int*)out = sqrtl(*(const long int*)a);
        break;

    case NC_COMPLEX:
        *(double complex*)out = csqrt(*(const double complex*)a);
        break;

    default: break;
    }
}

NDArray* nc_sqrt(const NDArray* a, NDArray* out)
{
    nc_apply_unary(a, out, sqrt_worker);
    return out;
}


static void exp_worker(void* out, const void* a, DataType dtype)
{
    switch (dtype)
    {
    case NC_INT:
        *(int*)out = exp(*(const int*)a);
        break;

    case NC_FLOAT:
        *(float*)out = expf(*(const float*)a);
        break;

    case NC_DOUBLE:
        *(double*)out = exp(*(const double*)a);
        break;
    case NC_LONGINT:
        *(long int*)out = expl(*(const long int*)a);
        break;

    case NC_COMPLEX:
        *(double complex*)out = cexp(*(const double complex*)a);
        break;

    default: break;
    }
}


NDArray* nc_exp(const NDArray* a, NDArray* out)
{
    nc_apply_unary(a, out, exp_worker);
    return out;
}


static void log_worker(void* out, const void* a, DataType dtype)
{
    switch (dtype)
    {
    case NC_INT:
        *(int*)out = log(*(const int*)a);
        break;

    case NC_FLOAT:
        *(float*)out = log(*(const float*)a);
        break;

    case NC_DOUBLE:
        *(double*)out = log(*(const double*)a);
        break;
    case NC_LONGINT:
        *(long int*)out = log(*(const long int*)a);
        break;

    case NC_COMPLEX:
        *(double complex*)out = clog(*(const double complex*)a);
        break;

    default: break;
    }
}

NDArray* nc_log(const NDArray* a, NDArray* out)
{
    nc_apply_unary(a, out, log_worker);
    return out;
}

static void log10_worker(void* out, const void* a, DataType dtype)
{
    switch (dtype)
    {
    case NC_INT:
        *(int*)out = log10(*(const int*)a);
        break;

    case NC_FLOAT:
        *(float*)out = log10(*(const float*)a);
        break;

    case NC_DOUBLE:
        *(double*)out = log10(*(const double*)a);
        break;
    case NC_LONGINT:
        *(long int*)out = log10(*(const long int*)a);
        break;

    case NC_COMPLEX:
        *(double complex*)out = clog(*(const double complex*)a) / log(10);
        break;

    default: break;
    }
}

NDArray* nc_log10(const NDArray* a, NDArray* out)
{
    nc_apply_unary(a, out, log10_worker);
    return out;
}

DEFINE_UNARY(sin, sin);

NDArray* nc_sin(const NDArray* a, NDArray* out)
{
    nc_apply_unary(a, out, sin_worker);
    return out;
}

DEFINE_UNARY(cos, cos);

NDArray* nc_cos(const NDArray* a, NDArray* out)
{
    nc_apply_unary(a, out, cos_worker);
    return out;
}

DEFINE_UNARY(tan, tan);

NDArray* nc_tan(const NDArray* a, NDArray* out)
{
    nc_apply_unary(a, out, tan_worker);
    return out;
}


static void eq_worker(void* out, const void* a, const void* b, DataType dtype)
{
    switch (dtype)
    {
    case NC_INT:
        *(int*)out = *(const int*)a == *(const int*)b;
        break;
    case NC_FLOAT:
        *(int*)out = nc_isclose_float(*(const float*)a, *(const float*)b);
        break;
    case NC_DOUBLE:
        *(int*)out = nc_isclose_double(*(const double*)a, *(const double*)b);
        break;
    case NC_LONGINT:
        *(int*)out = *(const long int*)a == *(const long int*)b;
        break;

    case NC_COMPLEX:
        *(int*)out = nc_isclose_complex(*(const double complex*)a, *(const double complex*)b);
        break;
    default:
        break;
    }
}


NDArray* nc_equal(const NDArray* a, const NDArray* b, NDArray* out)
{
    nc_apply_binary(a, b, out, eq_worker);
    return out;
}

static void not_eq_worker(void* out, const void* a, const void* b, DataType dtype)
{
    int result = 0;
    eq_worker(&result, a, b, dtype);
    *(int*)out = !result;
}

NDArray* nc_not_equal(const NDArray* a, const NDArray* b, NDArray* out)
{
    nc_apply_binary(a, b, out, not_eq_worker);
    return out;
}

#define DEFINE_BINARY_CLOGICAL_WORKER(name, op) \
static void name##_worker(void* out, const void* a, const void* b, DataType dtype) { \
switch (dtype) { \
case NC_INT:    *(int*)out = *(const int*)a op *(const int*)b; break; \
case NC_FLOAT:  *(int*)out = *(const float*)a op *(const float*)b; break; \
case NC_DOUBLE: *(int*)out = *(const double*)a op *(const double*)b; break; \
default: break; \
} \
}

DEFINE_BINARY_CLOGICAL_WORKER(gt,>)
NDArray* nc_greater(const NDArray* a, const NDArray* b, NDArray* out)
{
    nc_apply_binary(a, b, out, gt_worker);
    return out;
}


DEFINE_BINARY_CLOGICAL_WORKER(gte,>=)
NDArray* nc_greater_equal(const NDArray* a, const NDArray* b, NDArray* out)
{
    nc_apply_binary(a, b, out, gte_worker);
    return out;
}


DEFINE_BINARY_CLOGICAL_WORKER(lt,<)
NDArray* nc_less(const NDArray* a, const NDArray* b, NDArray* out)
{
    nc_apply_binary(a, b, out, lt_worker);
    return out;
}

DEFINE_BINARY_CLOGICAL_WORKER(le,<=)
NDArray* nc_less_equal(const NDArray* a, const NDArray* b, NDArray* out)
{
    nc_apply_binary(a, b, out, le_worker);
    return out;
}

static int nc_is_truthy(const void* value, DataType dtype)
{
    switch (dtype)
    {
    case NC_INT:
        return *(const int*)value != 0;
    case NC_FLOAT:
        return *(const float*)value != 0.0f;
    case NC_DOUBLE:
        return *(const double*)value != 0.0;
    case NC_LONGINT:
        return *(const long int*)value != 0;
    case NC_COMPLEX:
        return cabs(*(const double complex*)value) != 0.0;
    default:
        return 0;
    }
}

static void logical_and_worker(void* out, const void* a, const void* b, DataType dtype)
{
    *(int*)out = nc_is_truthy(a, dtype) && nc_is_truthy(b, dtype);
}

NDArray* nc_logical_and(const NDArray* a, const NDArray* b, NDArray* out)
{
    nc_apply_binary(a, b, out, logical_and_worker);
    return out;
}


static void logical_or_worker(void* out, const void* a, const void* b, DataType dtype)
{
    *(int*)out = nc_is_truthy(a, dtype) || nc_is_truthy(b, dtype);
}

NDArray* nc_logical_or(const NDArray* a, const NDArray* b, NDArray* out)
{
    nc_apply_binary(a, b, out, logical_or_worker);
    return out;
}

static void logical_not_worker(void* out, const void* a, DataType dtype)
{
    *(int*)out = !nc_is_truthy(a, dtype);
}

NDArray* nc_logical_not(const NDArray* a, NDArray* out)
{
    nc_apply_unary(a, out, logical_not_worker);
    return out;
}

typedef enum
{
    NC_REDUCE_SUM,
    NC_REDUCE_MEAN,
    NC_REDUCE_MAX,
    NC_REDUCE_MIN,
    NC_REDUCE_PROD
} NCReduceKind;

static NDArray* nc_reduce_prepare_output(const NDArray* a, int axis, bool keepdims)
{
    int out_ndim = 0;
    size_t* out_shape = NULL;
    NDArray* out;

    if (axis < -1 || axis >= a->ndim) return NULL;

    if (axis == -1)
    {
        if (keepdims)
        {
            out_ndim = a->ndim;
            if (out_ndim > 0)
            {
                out_shape = (size_t*)malloc((size_t)out_ndim * sizeof(size_t));
                if (!out_shape) return NULL;
                for (int i = 0; i < out_ndim; ++i) out_shape[i] = 1;
            }
        }
    }
    else if (keepdims)
    {
        out_ndim = a->ndim;
        out_shape = (size_t*)malloc((size_t)out_ndim * sizeof(size_t));
        if (!out_shape) return NULL;
        for (int i = 0; i < out_ndim; ++i)
        {
            out_shape[i] = (i == axis) ? 1 : a->shape[i];
        }
    }
    else
    {
        out_ndim = a->ndim - 1;
        if (out_ndim > 0)
        {
            int out_axis = 0;
            out_shape = (size_t*)malloc((size_t)out_ndim * sizeof(size_t));
            if (!out_shape) return NULL;
            for (int i = 0; i < a->ndim; ++i)
            {
                if (i == axis) continue;
                out_shape[out_axis++] = a->shape[i];
            }
        }
    }

    out = nc_array_empty(out_ndim, out_shape, a->dtype);
    free(out_shape);
    return out;
}

static size_t nc_reduce_output_index(
    const NDArray* a,
    const NDArray* out,
    const size_t* coords,
    int axis,
    bool keepdims
)
{
    size_t out_index = 0;
    int out_axis = 0;

    if (axis == -1) return 0;

    if (keepdims)
    {
        for (int i = 0; i < a->ndim; ++i)
        {
            const size_t coord = (i == axis) ? 0 : coords[i];
            out_index = out_index * out->shape[i] + coord;
        }
        return out_index;
    }

    for (int i = 0; i < a->ndim; ++i)
    {
        if (i == axis) continue;
        out_index = out_index * out->shape[out_axis++] + coords[i];
    }
    return out_index;
}

static NDArray* nc_reduce(const NDArray* a, int axis, bool keepdims, NCReduceKind kind)
{
    size_t* coords;
    uint8_t* seen = NULL;
    NDArray* out;
    size_t reduce_count;
    const size_t out_item_size = nc_dtype_size(a->dtype);

    if (!a || !a->data) return NULL;

    out = nc_reduce_prepare_output(a, axis, keepdims);
    if (!out) return NULL;

    if (out->size == 0) return out;

    const uint8_t* in_base = (const uint8_t*)a->data + a->offset;
    uint8_t* out_base = (uint8_t*)out->data + out->offset;

    if (kind == NC_REDUCE_MAX || kind == NC_REDUCE_MIN)
    {
        seen = (uint8_t*)calloc(out->size, sizeof(uint8_t));
        if (!seen)
        {
            nc_array_free(out);
            return NULL;
        }
    }
    else
    {
        for (size_t i = 0; i < out->size; ++i)
        {
            uint8_t* out_ptr = out_base + i * out_item_size;
            switch (a->dtype)
            {
            case NC_INT:
                *(int*)out_ptr = (kind == NC_REDUCE_PROD) ? 1 : 0;
                break;
            case NC_FLOAT:
                *(float*)out_ptr = (kind == NC_REDUCE_PROD) ? 1.0f : 0.0f;
                break;
            case NC_DOUBLE:
                *(double*)out_ptr = (kind == NC_REDUCE_PROD) ? 1.0 : 0.0;
                break;
            case NC_LONGINT:
                *(long int*)out_ptr = (kind == NC_REDUCE_PROD) ? 1L : 0L;
                break;
            case NC_COMPLEX:
                *(double complex*)out_ptr = (kind == NC_REDUCE_PROD) ? (1.0 + 0.0 * I) : (0.0 + 0.0 * I);
                break;
            default:
                break;
            }
        }
    }

    coords = (size_t*)malloc((size_t)a->ndim * sizeof(size_t));
    if (!coords)
    {
        free(seen);
        nc_array_free(out);
        return NULL;
    }

    for (size_t flat = 0; flat < a->size; ++flat)
    {
        size_t rem = flat;
        ptrdiff_t in_byte_offset = 0;
        size_t out_index;
        uint8_t* out_ptr;
        const uint8_t* in_ptr;

        for (int dim = a->ndim - 1; dim >= 0; --dim)
        {
            coords[dim] = rem % a->shape[dim];
            rem /= a->shape[dim];
            in_byte_offset += (ptrdiff_t)coords[dim] * a->strides[dim];
            if (dim == 0) break;
        }

        out_index = nc_reduce_output_index(a, out, coords, axis, keepdims);
        out_ptr = out_base + out_index * out_item_size;
        in_ptr = in_base + in_byte_offset;

        if (kind == NC_REDUCE_MAX || kind == NC_REDUCE_MIN)
        {
            if (!seen[out_index])
            {
                seen[out_index] = 1;
                switch (a->dtype)
                {
                case NC_INT:
                    *(int*)out_ptr = *(const int*)in_ptr;
                    break;
                case NC_FLOAT:
                    *(float*)out_ptr = *(const float*)in_ptr;
                    break;
                case NC_DOUBLE:
                    *(double*)out_ptr = *(const double*)in_ptr;
                    break;
                case NC_LONGINT:
                    *(long int*)out_ptr = *(const long int*)in_ptr;
                    break;
                case NC_COMPLEX:
                    *(double complex*)out_ptr = *(const double complex*)in_ptr;
                    break;
                default:
                    break;
                }
                continue;
            }
        }

        switch (a->dtype)
        {
        case NC_INT:
            if (kind == NC_REDUCE_SUM || kind == NC_REDUCE_MEAN) *(int*)out_ptr += *(const int*)in_ptr;
            else if (kind == NC_REDUCE_PROD) *(int*)out_ptr *= *(const int*)in_ptr;
            else if (kind == NC_REDUCE_MAX && *(const int*)in_ptr > *(int*)out_ptr) *(int*)out_ptr = *(const int*)in_ptr;
            else if (kind == NC_REDUCE_MIN && *(const int*)in_ptr < *(int*)out_ptr) *(int*)out_ptr = *(const int*)in_ptr;
            break;
        case NC_FLOAT:
            if (kind == NC_REDUCE_SUM || kind == NC_REDUCE_MEAN) *(float*)out_ptr += *(const float*)in_ptr;
            else if (kind == NC_REDUCE_PROD) *(float*)out_ptr *= *(const float*)in_ptr;
            else if (kind == NC_REDUCE_MAX && *(const float*)in_ptr > *(float*)out_ptr) *(float*)out_ptr = *(const float*)in_ptr;
            else if (kind == NC_REDUCE_MIN && *(const float*)in_ptr < *(float*)out_ptr) *(float*)out_ptr = *(const float*)in_ptr;
            break;
        case NC_DOUBLE:
            if (kind == NC_REDUCE_SUM || kind == NC_REDUCE_MEAN) *(double*)out_ptr += *(const double*)in_ptr;
            else if (kind == NC_REDUCE_PROD) *(double*)out_ptr *= *(const double*)in_ptr;
            else if (kind == NC_REDUCE_MAX && *(const double*)in_ptr > *(double*)out_ptr) *(double*)out_ptr = *(const double*)in_ptr;
            else if (kind == NC_REDUCE_MIN && *(const double*)in_ptr < *(double*)out_ptr) *(double*)out_ptr = *(const double*)in_ptr;
            break;
        case NC_LONGINT:
            if (kind == NC_REDUCE_SUM || kind == NC_REDUCE_MEAN) *(long int*)out_ptr += *(const long int*)in_ptr;
            else if (kind == NC_REDUCE_PROD) *(long int*)out_ptr *= *(const long int*)in_ptr;
            else if (kind == NC_REDUCE_MAX && *(const long int*)in_ptr > *(long int*)out_ptr) *(long int*)out_ptr = *(const long int*)in_ptr;
            else if (kind == NC_REDUCE_MIN && *(const long int*)in_ptr < *(long int*)out_ptr) *(long int*)out_ptr = *(const long int*)in_ptr;
            break;
        case NC_COMPLEX:
            if (kind == NC_REDUCE_SUM || kind == NC_REDUCE_MEAN) *(double complex*)out_ptr += *(const double complex*)in_ptr;
            else if (kind == NC_REDUCE_PROD) *(double complex*)out_ptr *= *(const double complex*)in_ptr;
            else if (kind == NC_REDUCE_MAX && cabs(*(const double complex*)in_ptr) > cabs(*(double complex*)out_ptr)) *(double complex*)out_ptr = *(const double complex*)in_ptr;
            else if (kind == NC_REDUCE_MIN && cabs(*(const double complex*)in_ptr) < cabs(*(double complex*)out_ptr)) *(double complex*)out_ptr = *(const double complex*)in_ptr;
            break;
        default:
            break;
        }
    }

    free(coords);
    free(seen);

    if (kind != NC_REDUCE_MEAN) return out;

    reduce_count = (axis == -1) ? a->size : a->shape[axis];
    if (reduce_count == 0) return out;

    for (size_t i = 0; i < out->size; ++i)
    {
        uint8_t* out_ptr = out_base + i * out_item_size;
        switch (a->dtype)
        {
        case NC_INT:
            *(int*)out_ptr /= (int)reduce_count;
            break;
        case NC_FLOAT:
            *(float*)out_ptr /= (float)reduce_count;
            break;
        case NC_DOUBLE:
            *(double*)out_ptr /= (double)reduce_count;
            break;
        case NC_LONGINT:
            *(long int*)out_ptr /= (long int)reduce_count;
            break;
        case NC_COMPLEX:
            *(double complex*)out_ptr /= (double)reduce_count;
            break;
        default:
            break;
        }
    }

    return out;
}

NDArray* nc_sum(const NDArray* a, int axis, bool keepdims)
{
    return nc_reduce(a, axis, keepdims, NC_REDUCE_SUM);
}

NDArray* nc_mean(const NDArray* a, int axis, bool keepdims)
{
    return nc_reduce(a, axis, keepdims, NC_REDUCE_MEAN);
}

NDArray* nc_max(const NDArray* a, int axis, bool keepdims)
{
    return nc_reduce(a, axis, keepdims, NC_REDUCE_MAX);
}

NDArray* nc_min(const NDArray* a, int axis, bool keepdims)
{
    return nc_reduce(a, axis, keepdims, NC_REDUCE_MIN);
}

NDArray* nc_prod(const NDArray* a, int axis, bool keepdims)
{
    return nc_reduce(a, axis, keepdims, NC_REDUCE_PROD);
}

static const uint8_t* nc_ptr_at_coords_const(const NDArray* a, const size_t* coords)
{
    ptrdiff_t byte_offset = a->offset;
    for (int i = 0; i < a->ndim; ++i)
    {
        byte_offset += (ptrdiff_t)coords[i] * a->strides[i];
    }
    return (const uint8_t*)a->data + byte_offset;
}

static uint8_t* nc_ptr_at_coords(NDArray* a, const size_t* coords)
{
    ptrdiff_t byte_offset = a->offset;
    for (int i = 0; i < a->ndim; ++i)
    {
        byte_offset += (ptrdiff_t)coords[i] * a->strides[i];
    }
    return (uint8_t*)a->data + byte_offset;
}

static double complex nc_read_as_complex(const uint8_t* ptr, DataType dtype)
{
    switch (dtype)
    {
    case NC_INT:
        return (double)(*(const int*)ptr) + 0.0 * I;
    case NC_FLOAT:
        return (double)(*(const float*)ptr) + 0.0 * I;
    case NC_DOUBLE:
        return *(const double*)ptr + 0.0 * I;
    case NC_LONGINT:
        return (double)(*(const long int*)ptr) + 0.0 * I;
    case NC_COMPLEX:
        return *(const double complex*)ptr;
    default:
        return 0.0 + 0.0 * I;
    }
}

static void nc_write_from_complex(uint8_t* ptr, DataType dtype, double complex value)
{
    switch (dtype)
    {
    case NC_INT:
        *(int*)ptr = (int)creal(value);
        break;
    case NC_FLOAT:
        *(float*)ptr = (float)creal(value);
        break;
    case NC_DOUBLE:
        *(double*)ptr = creal(value);
        break;
    case NC_LONGINT:
        *(long int*)ptr = (long int)creal(value);
        break;
    case NC_COMPLEX:
        *(double complex*)ptr = value;
        break;
    default:
        break;
    }
}

static NDArray* nc_matmul_2d(const NDArray* a, const NDArray* b)
{
    if (!a || !b) return NULL;
    if (a->ndim != 2 || b->ndim != 2) return NULL;
    if (a->dtype != b->dtype) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;

    size_t out_shape[2] = {a->shape[0], b->shape[1]};
    NDArray* out = nc_array_zeros(2, out_shape, a->dtype);
    if (!out) return NULL;

    for (size_t i = 0; i < out_shape[0]; ++i)
    {
        for (size_t j = 0; j < out_shape[1]; ++j)
        {
            double complex acc = 0.0 + 0.0 * I;
            for (size_t k = 0; k < a->shape[1]; ++k)
            {
                const size_t a_coords[2] = {i, k};
                const size_t b_coords[2] = {k, j};
                const double complex av = nc_read_as_complex(nc_ptr_at_coords_const(a, a_coords), a->dtype);
                const double complex bv = nc_read_as_complex(nc_ptr_at_coords_const(b, b_coords), b->dtype);
                acc += av * bv;
            }
            const size_t out_coords[2] = {i, j};
            nc_write_from_complex(nc_ptr_at_coords(out, out_coords), out->dtype, acc);
        }
    }

    return out;
}

NDArray* nc_dot(const NDArray* a, const NDArray* b)
{
    if (!a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;

    if (a->ndim == 1 && b->ndim == 1)
    {
        if (a->shape[0] != b->shape[0]) return NULL;
        NDArray* out = nc_array_zeros(0, NULL, a->dtype);
        if (!out) return NULL;

        double complex acc = 0.0 + 0.0 * I;
        for (size_t i = 0; i < a->shape[0]; ++i)
        {
            const size_t coords[1] = {i};
            const double complex av = nc_read_as_complex(nc_ptr_at_coords_const(a, coords), a->dtype);
            const double complex bv = nc_read_as_complex(nc_ptr_at_coords_const(b, coords), b->dtype);
            acc += av * bv;
        }
        nc_write_from_complex((uint8_t*)out->data + out->offset, out->dtype, acc);
        return out;
    }

    if (a->ndim == 2 && b->ndim == 2)
    {
        return nc_matmul_2d(a, b);
    }

    if (a->ndim == 2 && b->ndim == 1)
    {
        if (a->shape[1] != b->shape[0]) return NULL;
        size_t out_shape[1] = {a->shape[0]};
        NDArray* out = nc_array_zeros(1, out_shape, a->dtype);
        if (!out) return NULL;

        for (size_t i = 0; i < out_shape[0]; ++i)
        {
            double complex acc = 0.0 + 0.0 * I;
            for (size_t k = 0; k < a->shape[1]; ++k)
            {
                const size_t a_coords[2] = {i, k};
                const size_t b_coords[1] = {k};
                const double complex av = nc_read_as_complex(nc_ptr_at_coords_const(a, a_coords), a->dtype);
                const double complex bv = nc_read_as_complex(nc_ptr_at_coords_const(b, b_coords), b->dtype);
                acc += av * bv;
            }
            const size_t out_coords[1] = {i};
            nc_write_from_complex(nc_ptr_at_coords(out, out_coords), out->dtype, acc);
        }
        return out;
    }

    if (a->ndim == 1 && b->ndim == 2)
    {
        if (a->shape[0] != b->shape[0]) return NULL;
        size_t out_shape[1] = {b->shape[1]};
        NDArray* out = nc_array_zeros(1, out_shape, a->dtype);
        if (!out) return NULL;

        for (size_t j = 0; j < out_shape[0]; ++j)
        {
            double complex acc = 0.0 + 0.0 * I;
            for (size_t k = 0; k < a->shape[0]; ++k)
            {
                const size_t a_coords[1] = {k};
                const size_t b_coords[2] = {k, j};
                const double complex av = nc_read_as_complex(nc_ptr_at_coords_const(a, a_coords), a->dtype);
                const double complex bv = nc_read_as_complex(nc_ptr_at_coords_const(b, b_coords), b->dtype);
                acc += av * bv;
            }
            const size_t out_coords[1] = {j};
            nc_write_from_complex(nc_ptr_at_coords(out, out_coords), out->dtype, acc);
        }
        return out;
    }

    return NULL;
}

NDArray* nc_matmul(const NDArray* a, const NDArray* b)
{
    if (!a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;

    if (a->ndim == 2 && b->ndim == 2) return nc_matmul_2d(a, b);
    if (a->ndim == 1 || b->ndim == 1) return nc_dot(a, b);
    return NULL;
}

NDArray* nc_transpose(const NDArray* a)
{
    NDArray* out;

    if (!a) return NULL;
    out = (NDArray*)calloc(1, sizeof(NDArray));
    if (!out) return NULL;

    out->data = a->data;
    out->ndim = a->ndim;
    out->size = a->size;
    out->dtype = a->dtype;
    out->offset = a->offset;

    if (a->ndim == 0) return out;

    out->shape = (size_t*)malloc((size_t)a->ndim * sizeof(size_t));
    out->strides = (int*)malloc((size_t)a->ndim * sizeof(int));
    if (!out->shape || !out->strides)
    {
        free(out->shape);
        free(out->strides);
        free(out);
        return NULL;
    }

    for (int i = 0; i < a->ndim; ++i)
    {
        out->shape[i] = a->shape[a->ndim - 1 - i];
        out->strides[i] = a->strides[a->ndim - 1 - i];
    }

    return out;
}
