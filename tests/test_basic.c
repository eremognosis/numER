#include <stdio.h>
#include <assert.h>
#include <numER/numc.h>
#include <numER/broadcasting.h>
#include <stdlib.h>

int main() {
    printf("Running basic array allocation test...\n");

    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3};
    size_t shape_c[] = {2, 2};

    NDArray* a = nc_array_zeros(2, shape_a, NC_INT);
    NDArray* b = nc_array_zeros(1, shape_b, NC_INT);
    NDArray* c = nc_array_zeros(2, shape_c, NC_INT);
    assert(a && b && c);

    NDArray* ok_arrays[] = {a, b};
    NDArray* bad_arrays[] = {a, c};

    assert(nc_broadcast_check(2, ok_arrays) == 1);
    assert(nc_broadcast_check(2, bad_arrays) == 0);

    int out_ndim = -1;
    size_t* out_shape = NULL;
    assert(nc_compute_broadcast_shape(2, ok_arrays, &out_ndim, &out_shape) == 1);
    assert(out_ndim == 2);
    assert(out_shape);
    assert(out_shape[0] == 2);
    assert(out_shape[1] == 3);

    int* bcast_strides = nc_compute_broadcast_strides(b, out_ndim, out_shape);
    assert(bcast_strides);
    assert(bcast_strides[0] == 0);
    assert(bcast_strides[1] == b->strides[0]);

    free(bcast_strides);
    free(out_shape);
    nc_array_free(a);
    nc_array_free(b);
    nc_array_free(c);

    return 0;
}
