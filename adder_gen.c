#define NN_IMPLEMENTATION
#include "nn.h"

#define BITS 4

int main(void)
{
    size_t n = (1<<BITS);
    size_t rows = n*n;
    Mat t  = mat_alloc(rows, 2*BITS + BITS + 1);
    Mat ti = {
        .es = &MAT_AT(t, 0, 0),
        .rows = t.rows,
        .cols = 2*BITS,
        .stride = t.stride,
    };
    Mat to = {
        .es = &MAT_AT(t, 0, 2*BITS),
        .rows = t.rows,
        .cols = BITS + 1,
        .stride = t.stride,
    };
    for (size_t i = 0; i < ti.rows; ++i) {
        size_t x = i/n;
        size_t y = i%n;
        size_t z = x + y;
        for (size_t j = 0; j < BITS; ++j) {
            MAT_AT(ti, i, j)        = (x>>j)&1;
            MAT_AT(ti, i, j + BITS) = (y>>j)&1;
            MAT_AT(to, i, j)        = (z>>j)&1;
        }
        MAT_AT(to, i, BITS) = z >= n;
    }
    const char *out_file_path = "adder.mat";
    FILE *out = fopen(out_file_path, "wb");
    if (out == NULL) {
        fprintf(stderr, "ERROR: could not open file %s\n", out_file_path);
        return 1;
    }
    mat_save(out, t);
    fclose(out);
    printf("Generated %s\n", out_file_path);
    return 0;
}
