#ifndef BROADCASTING_H_
#define BROADCASTING_H_

#include "array.h"
#include <stdint.h>
/**
 * Validates if a set of arrays can be broadcast together.
 * Follows the "Right-to-Left" alignment rule.
 * 
 * @return 1 if compatible, 0 if not. technically bool but i felt like keeping int
 */
int nc_broadcast_check(int n_arrays, NDArray** arrays);

/**
 * Calculates the resulting shape of a broadcast operation.
 * 
 * @param n_arrays Number of input arrays
 * @param arrays Array of NDArray pointers
 * @param out_ndim Pointer to store the resulting number of dimensions
 * @param out_shape Pointer to an allocated array of size_t (must be freed by caller or let the memory say this to you)
 * @return 1 on success, 0 on failure (incompatible shapes)
 */
int nc_compute_broadcast_shape(int n_arrays, NDArray** arrays, int* out_ndim, size_t** out_shape);

/**
 * Generates the "Fake Strides" for a source array to match a target shape.
 * This is the magic: if an axis is broadcasted (size 1 -> size N), 
 * the stride for that axis is set to 0.
 * 
 * @param arr The source array
 * @param target_ndim The ndim of the broadcasted result
 * @param target_shape The shape of the broadcasted result
 * @return An array of strides (size target_ndim) or NULL on error.
 */
int* nc_compute_broadcast_strides(const NDArray* arr, int target_ndim, const size_t* target_shape);

#endif