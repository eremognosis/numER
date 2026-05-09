#ifndef ITERATOR_HPP_

#define ITERATOR_HPP_
#include <stddef.h>
#include <stdint.h>
#include "numER/array.h"

typedef struct
{
    int ndim; /// Number of dimensions being iterated
    size_t* shape; /// The broadcasted shape
    size_t size; /// Total number of elements to visit
    size_t index; // Current flat index (0 to size-1)

    // Arrays of size 'ndim' for tracking progress
    size_t* coordinates; // Current multi-dimensional index (e.g., [1, 0, 3])

    // Pointers and Strides for each operand (e.g., 2 for binary ops)
    int n_arrays;
    uint8_t** data_ptrs; // Current pointer for each array
    int** strides; // Strides for each array (broadcast-aware)
    int** backstrides; // "Reset" values to jump back to start of an axis
} NCIterator;


/** Creates an iterator for N arrays. Handles broadcasting internally.
 *
 * @param n_arrays
 * @param arrays
 * If shapes are incompatible, returns NULL.
 */
NCIterator* nc_iter_new(int n_arrays, NDArray** arrays);


/** Moves the data_ptrs to the next logical element based on strides.
 *
 * @param iter The iterator structor
 * @return Returns 1 if successful, 0 if we've reached the end.
 */
int nc_iter_next(NCIterator *iter);

/**
 *
 * @param iter
 * @param array_index
 * @return
 */
void* nc_iter_get_ptr(NCIterator *iter, int array_index);


/** Cleanup
 *
 * @param iter cleans it
 */
void nc_iter_free(NCIterator *iter);
#endif
