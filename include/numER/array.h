//
// Created by raj on 5/9/26.
//

#ifndef NUMER_ARRAY_H
#define NUMER_ARRAY_H
#include <stdlib.h>


typedef  enum
{
    FLOAT,
    INT,
    COMPLEX,
} DataType;


/**
 * The main n-dimensional array object
 * 
 */
typedef struct {
    void *data;          // Pointer to the start of the data block
    int ndim;            // Number of dimensions
    size_t *shape;       // Array of size ndim (e.g., [3, 4])
    int *strides;        // Array of size ndim (byte jumps per axis)
    size_t size;         // Total number of elements
    DataType dtype;      // Enum for FLOAT, INT, etc.
    int offset;          // Pointer offset for slicing
} NDArray;



#endif //NUMER_ARRAY_H
