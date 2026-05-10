#include "numER/ops.h"
#include "numER/iterator.h"
#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>
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


static  void log_worker(void* out, const void* a, DataType dtype)
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

static  void log10_worker(void* out, const void* a, DataType dtype)
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
        *(double complex*)out = clog(*(const double complex*)a)/log(10);
        break;

    default: break;
    }
}
NDArray* nc_log10(const NDArray* a, NDArray* out)
{
    nc_apply_unary(a, out, log10_worker);
    return out;
}

DEFINE_UNARY(sin,sin);
NDArray* nc_sin(const NDArray* a, NDArray* out)
{
    nc_apply_unary(a, out, sin_worker);
    return out;
}

DEFINE_UNARY(cos,cos);
NDArray* nc_cos(const NDArray* a, NDArray* out)
{
    nc_apply_unary(a, out, cos_worker);
    return out;
}

DEFINE_UNARY(tan,tan);
NDArray* nc_tan(const NDArray* a, NDArray* out)
{
    nc_apply_unary(a, out, tan_worker);
    return out;
}
