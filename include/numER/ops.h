#ifndef OPS_H_
#define OPS_H_

#include "array.h"

/**
 * @brief THE GRAND UNIFIED UFUNC PATTERN
 * * Every binary op follows: out = op(a, b)
 * If 'out' is NULL, a new array is allocated with the broadcasted shape.
 * If 'out' is provided, it must be broadcast-compatible with a and b.
 */

// ============================================================================
// Basic Arithmetic (The "I could have done this in Python" section)
// ============================================================================


/**
 *
 * @param a First Array
 * @param b Second Array
 * @param out Output Array
 * @return Output array after populating
 */
NDArray* nc_add(const NDArray* a, const NDArray* b, NDArray* out);
NDArray* nc_subtract(const NDArray* a, const NDArray* b, NDArray* out);
NDArray* nc_multiply(const NDArray* a, const NDArray* b, NDArray* out);
NDArray* nc_divide(const NDArray* a, const NDArray* b, NDArray* out);
NDArray* nc_power(const NDArray* a, const NDArray* b, NDArray* out);
NDArray* nc_remainder(const NDArray* a, const NDArray* b, NDArray* out);

// ============================================================================
// Unary Operations (One-man army functions)
// ============================================================================

NDArray* nc_negative(const NDArray* a, NDArray* out);
NDArray* nc_absolute(const NDArray* a, NDArray* out);
NDArray* nc_sqrt(const NDArray* a, NDArray* out);
NDArray* nc_exp(const NDArray* a, NDArray* out);
NDArray* nc_log(const NDArray* a, NDArray* out);
NDArray* nc_log10(const NDArray* a, NDArray* out);

/**
 *
 * @param a the array
 * @param out thje array
 * @return sin/cos/tan in radian
 */
NDArray* nc_sin(const NDArray* a, NDArray* out);
NDArray* nc_cos(const NDArray* a, NDArray* out);
NDArray* nc_tan(const NDArray* a, NDArray* out);

// ================= ===========================================================
// Comparison and Logic (For your rationalist skepticism)
// ============================================================================

// These return an array of NC_INT (0 or 1) representing booleans
NDArray* nc_equal(const NDArray* a, const NDArray* b, NDArray* out);
NDArray* nc_not_equal(const NDArray* a, const NDArray* b, NDArray* out);
NDArray* nc_greater(const NDArray* a, const NDArray* b, NDArray* out);
NDArray* nc_greater_equal(const NDArray* a, const NDArray* b, NDArray* out);
NDArray* nc_less(const NDArray* a, const NDArray* b, NDArray* out);
NDArray* nc_less_equal(const NDArray* a, const NDArray* b, NDArray* out);

NDArray* nc_logical_and(const NDArray* a, const NDArray* b, NDArray* out);
NDArray* nc_logical_or(const NDArray* a, const NDArray* b, NDArray* out);
NDArray* nc_logical_not(const NDArray* a, NDArray* out);

// ============================================================================
// Reductions
// ============================================================================

/**
 * @param axis The axis to reduce. If -1, reduces the entire array to a scalar.
 * @param keepdims If true, the reduced axes are left in the result as dimensions with size one.
 */
NDArray* nc_sum(const NDArray* a, int axis, bool keepdims);
NDArray* nc_mean(const NDArray* a, int axis, bool keepdims);
NDArray* nc_max(const NDArray* a, int axis, bool keepdims);
NDArray* nc_min(const NDArray* a, int axis, bool keepdims);
NDArray* nc_prod(const NDArray* a, int axis, bool keepdims);

// ============================================================================
// Linear Algebra
// ============================================================================

// Standard dot product for vectors, matrix-vector, or matrix-matrix
NDArray* nc_dot(const NDArray* a, const NDArray* b);

// Proper matrix multiplication
NDArray* nc_matmul(const NDArray* a, const NDArray* b);

// The transpose view (swaps strides without copying data, very fast, very elegant)
NDArray* nc_transpose(const NDArray* a);


// ============================================================================
// Generic Apply
// ============================================================================

typedef void (*nc_unary_op)(void* out, const void* a, DataType dtype);
typedef void (*nc_binary_op)(void* out, const void* a, const void* b, DataType dtype);

/**
 * Internal engine that drives the ufuncs. Uses NCIterator.
 */
void nc_apply_unary(const NDArray* a, NDArray* out, nc_unary_op op);
void nc_apply_binary(const NDArray* a, const NDArray* b, NDArray* out, nc_binary_op op);
#endif // OPS_H_
