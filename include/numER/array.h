#ifndef ARRAY_HPP_
#define ARRAY_HPP_

#include <stddef.h>
#include <stdbool.h>

// The enyum to look cute
typedef enum
{
    NC_FLOAT, ///
    NC_INT, /// standard int
    NC_COMPLEX, // double complex
    NC_DOUBLE,
    NC_LONGINT
} DataType;

/** The main n-dimensional array object
 */
typedef struct
{
    void* data; // Pointer to the start of the data block
    int ndim; // Number of dimensions
    size_t* shape; // Array of size ndim (e.g., [3, 4])
    int* strides; // Array of size ndim (byte jumps per axis)
    size_t size; // Total number of elements
    DataType dtype; // Enum for FLOAT, INT, etc.
    int offset; // Pointer offset for slicing
} NDArray;

// ============================================================================
// Core Lifecycle (Allocation & Destruction)
// ============================================================================

/**
 * Returns the byte size of a given DataType.
 */
size_t nc_dtype_size(DataType dtype);

/**
 * Allocates an uninitialized array (equivalent to np.empty).
 * Computes the contiguous strides automatically.
 */
NDArray* nc_array_empty(int ndim, const size_t* shape, DataType dtype);

/**
 * Allocates an array initialized to zero (equivalent to np.zeros).
 */
NDArray* nc_array_zeros(int ndim, const size_t* shape, DataType dtype);

/**
 * Destroys the array. 
 * WARNING: Because NDArray lacks ownership metadata, calling this on a 
 * "view" will free the memory of the original array, leading to a Segfault 
 * the next time the original array is used. 
 */
void nc_array_free(NDArray* array);

// ============================================================================
// Views and Slicing
// ============================================================================

/**
 * Creates a shallow copy (a view) of the given array.
 * Allocates a new NDArray struct, but `data` points to the exact same block.
 */
NDArray* nc_array_view(const NDArray* source);

/**
 * Attempts to reshape the array without copying data.
 * Returns a new view with updated shape/strides if the array is contiguous.
 * Returns NULL if a copy would be required
 */
NDArray* nc_array_reshape(const NDArray* source, int new_ndim, const size_t* new_shape);

// ============================================================================
// Utilities
// ============================================================================

/**
 * Checks if the array is C-style contiguous.
 */
bool nc_array_is_contiguous(const NDArray* array);

/**
 * Prints the metadata (shape, strides, dtype) and the raw data.
 */
void nc_array_print(const NDArray* array);

#endif // ARRAY_HPP_
