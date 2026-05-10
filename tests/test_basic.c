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

    printf("Running eq/not_equal tolerance test...\n");
    size_t shape_cmp[] = {3};
    NDArray* lhs = nc_array_zeros(1, shape_cmp, NC_FLOAT);
    NDArray* rhs = nc_array_zeros(1, shape_cmp, NC_FLOAT);
    NDArray* eq_out = nc_array_zeros(1, shape_cmp, NC_INT);
    NDArray* ne_out = nc_array_zeros(1, shape_cmp, NC_INT);
    assert(lhs && rhs && eq_out && ne_out);

    float* lhs_data = (float*)lhs->data;
    float* rhs_data = (float*)rhs->data;
    int* eq_data = (int*)eq_out->data;
    int* ne_data = (int*)ne_out->data;

    lhs_data[0] = 1.0f; rhs_data[0] = 1.000005f; // close: within rtol
    lhs_data[1] = 2.0f; rhs_data[1] = 2.0001f;   // not close
    lhs_data[2] = 0.0f; rhs_data[2] = 0.000000005f; // close: within atol

    assert(nc_equal(lhs, rhs, eq_out) == eq_out);
    assert(eq_data[0] == 1);
    assert(eq_data[1] == 0);
    assert(eq_data[2] == 1);

    assert(nc_not_equal(lhs, rhs, ne_out) == ne_out);
    assert(ne_data[0] == 0);
    assert(ne_data[1] == 1);
    assert(ne_data[2] == 0);

    nc_array_free(lhs);
    nc_array_free(rhs);
    nc_array_free(eq_out);
    nc_array_free(ne_out);

    printf("Running logical and/or/not test...\n");
    NDArray* la = nc_array_zeros(1, shape_cmp, NC_INT);
    NDArray* lb = nc_array_zeros(1, shape_cmp, NC_INT);
    NDArray* and_out = nc_array_zeros(1, shape_cmp, NC_INT);
    NDArray* or_out = nc_array_zeros(1, shape_cmp, NC_INT);
    NDArray* not_out = nc_array_zeros(1, shape_cmp, NC_INT);
    assert(la && lb && and_out && or_out && not_out);

    int* la_data = (int*)la->data;
    int* lb_data = (int*)lb->data;
    int* and_data = (int*)and_out->data;
    int* or_data = (int*)or_out->data;
    int* not_data = (int*)not_out->data;

    la_data[0] = 0;  lb_data[0] = 0;
    la_data[1] = 2;  lb_data[1] = 0;
    la_data[2] = -3; lb_data[2] = 5;

    assert(nc_logical_and(la, lb, and_out) == and_out);
    assert(and_data[0] == 0);
    assert(and_data[1] == 0);
    assert(and_data[2] == 1);

    assert(nc_logical_or(la, lb, or_out) == or_out);
    assert(or_data[0] == 0);
    assert(or_data[1] == 1);
    assert(or_data[2] == 1);

    assert(nc_logical_not(la, not_out) == not_out);
    assert(not_data[0] == 1);
    assert(not_data[1] == 0);
    assert(not_data[2] == 0);

    nc_array_free(la);
    nc_array_free(lb);
    nc_array_free(and_out);
    nc_array_free(or_out);
    nc_array_free(not_out);

    printf("Running reductions test...\n");
    NDArray* r = nc_array_zeros(2, shape_a, NC_INT);
    assert(r);
    int* r_data = (int*)r->data;
    r_data[0] = 1; r_data[1] = 2; r_data[2] = 3;
    r_data[3] = 4; r_data[4] = 5; r_data[5] = 6;

    NDArray* sum_all = nc_sum(r, -1, false);
    assert(sum_all && sum_all->ndim == 0);
    assert(*(int*)sum_all->data == 21);

    NDArray* sum_axis0 = nc_sum(r, 0, false);
    assert(sum_axis0 && sum_axis0->ndim == 1 && sum_axis0->shape[0] == 3);
    assert(((int*)sum_axis0->data)[0] == 5);
    assert(((int*)sum_axis0->data)[1] == 7);
    assert(((int*)sum_axis0->data)[2] == 9);

    NDArray* sum_axis1_keep = nc_sum(r, 1, true);
    assert(sum_axis1_keep && sum_axis1_keep->ndim == 2);
    assert(sum_axis1_keep->shape[0] == 2 && sum_axis1_keep->shape[1] == 1);
    assert(((int*)sum_axis1_keep->data)[0] == 6);
    assert(((int*)sum_axis1_keep->data)[1] == 15);

    NDArray* max_axis1 = nc_max(r, 1, false);
    assert(max_axis1 && max_axis1->ndim == 1 && max_axis1->shape[0] == 2);
    assert(((int*)max_axis1->data)[0] == 3);
    assert(((int*)max_axis1->data)[1] == 6);

    NDArray* min_axis0 = nc_min(r, 0, false);
    assert(min_axis0 && min_axis0->ndim == 1 && min_axis0->shape[0] == 3);
    assert(((int*)min_axis0->data)[0] == 1);
    assert(((int*)min_axis0->data)[1] == 2);
    assert(((int*)min_axis0->data)[2] == 3);

    NDArray* prod_all = nc_prod(r, -1, false);
    assert(prod_all && prod_all->ndim == 0);
    assert(*(int*)prod_all->data == 720);

    size_t shape_d[] = {2, 2};
    NDArray* d = nc_array_zeros(2, shape_d, NC_DOUBLE);
    assert(d);
    double* d_data = (double*)d->data;
    d_data[0] = 1.0; d_data[1] = 2.0;
    d_data[2] = 3.0; d_data[3] = 4.0;

    NDArray* mean_all_keep = nc_mean(d, -1, true);
    assert(mean_all_keep && mean_all_keep->ndim == 2);
    assert(mean_all_keep->shape[0] == 1 && mean_all_keep->shape[1] == 1);
    assert(((double*)mean_all_keep->data)[0] == 2.5);

    NDArray* mean_axis0 = nc_mean(d, 0, false);
    assert(mean_axis0 && mean_axis0->ndim == 1 && mean_axis0->shape[0] == 2);
    assert(((double*)mean_axis0->data)[0] == 2.0);
    assert(((double*)mean_axis0->data)[1] == 3.0);

    nc_array_free(sum_all);
    nc_array_free(sum_axis0);
    nc_array_free(sum_axis1_keep);
    nc_array_free(max_axis1);
    nc_array_free(min_axis0);
    nc_array_free(prod_all);
    nc_array_free(d);
    nc_array_free(mean_all_keep);
    nc_array_free(mean_axis0);

    printf("Running linear algebra test...\n");
    size_t shape_v[] = {3};
    NDArray* v1 = nc_array_zeros(1, shape_v, NC_INT);
    NDArray* v2 = nc_array_zeros(1, shape_v, NC_INT);
    assert(v1 && v2);
    ((int*)v1->data)[0] = 1; ((int*)v1->data)[1] = 2; ((int*)v1->data)[2] = 3;
    ((int*)v2->data)[0] = 4; ((int*)v2->data)[1] = 5; ((int*)v2->data)[2] = 6;

    NDArray* dot_vv = nc_dot(v1, v2);
    assert(dot_vv && dot_vv->ndim == 0);
    assert(*(int*)dot_vv->data == 32);

    NDArray* mv = nc_dot(r, v1);
    assert(mv && mv->ndim == 1 && mv->shape[0] == 2);
    assert(((int*)mv->data)[0] == 14);
    assert(((int*)mv->data)[1] == 32);

    size_t shape_rt[] = {3, 2};
    NDArray* rt = nc_array_zeros(2, shape_rt, NC_INT);
    assert(rt);
    ((int*)rt->data)[0] = 1; ((int*)rt->data)[1] = 4;
    ((int*)rt->data)[2] = 2; ((int*)rt->data)[3] = 5;
    ((int*)rt->data)[4] = 3; ((int*)rt->data)[5] = 6;

    NDArray* vm = nc_dot(v1, rt);
    assert(vm && vm->ndim == 1 && vm->shape[0] == 2);
    assert(((int*)vm->data)[0] == 14);
    assert(((int*)vm->data)[1] == 32);

    NDArray* mm = nc_matmul(r, rt);
    assert(mm && mm->ndim == 2);
    assert(mm->shape[0] == 2 && mm->shape[1] == 2);
    assert(((int*)mm->data)[0] == 14);
    assert(((int*)mm->data)[1] == 32);
    assert(((int*)mm->data)[2] == 32);
    assert(((int*)mm->data)[3] == 77);

    NDArray* t = nc_transpose(r);
    assert(t && t->ndim == 2);
    assert(t->shape[0] == 3 && t->shape[1] == 2);
    assert(t->strides[0] == r->strides[1]);
    assert(t->strides[1] == r->strides[0]);
    assert(t->data == r->data);

    nc_array_free(v1);
    nc_array_free(v2);
    nc_array_free(dot_vv);
    nc_array_free(mv);
    nc_array_free(vm);
    nc_array_free(rt);
    nc_array_free(mm);
    nc_array_free(r);

    return 0;
}
