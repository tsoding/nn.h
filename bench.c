#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>
#include <stdio.h>

#define WARM_UP_TIME 3
#define ITERS 500

void mat_dot_old(Mat dst, Mat a, Mat b);

typedef void (*DotFunc)(Mat, Mat, Mat);
void bench(Mat dst, Mat a, Mat b, DotFunc func, char const *name);
void test_against(Mat a, Mat b, DotFunc reference, DotFunc to_test);

int main(void)
{
  /// setup
  size_t R = 300;
  size_t K = 200;
  size_t C = 400;
  Mat a = mat_alloc(R, K);
  Mat b = mat_alloc(K, C);
  Mat dst = mat_alloc(R, C);
  mat_rand(a, 0, 1);
  mat_rand(b, 0, 1);

  bench(dst, a, b, mat_dot_old, "old");
  bench(dst, a, b, mat_dot, "new");
  test_against(a, b, mat_dot_old, mat_dot);
}

void bench(Mat dst, Mat a, Mat b, DotFunc func, char const *name)
{
  double start = (double)clock() / CLOCKS_PER_SEC;
  double end = start;
  printf("Warming up for %d seconds...\n", WARM_UP_TIME);
  while (end-start < WARM_UP_TIME)
  {
    func(dst, a, b);
    end = (double)clock() / CLOCKS_PER_SEC;
  }
  printf("Running bench %s...\n", name);
  start = (double)clock() / CLOCKS_PER_SEC;
  for (size_t i = 0; i < ITERS; ++i)
  {
    func(dst, a, b);
  }
  end = (double)clock() / CLOCKS_PER_SEC;
  printf("%s solution took: %fs to process\n", name, end - start);
}

void mat_dot_old(Mat dst, Mat a, Mat b)
{
    NN_ASSERT(a.cols == b.rows);
    size_t n = a.cols;
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k < n; ++k) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

void test_against(Mat a, Mat b, DotFunc reference, DotFunc to_test) {
  Mat reference_res = mat_alloc(a.rows, b.cols);
  Mat test_res = mat_alloc(a.rows, b.cols);
  reference(reference_res, a, b);
  to_test(test_res, a, b);
  size_t total = reference_res.rows * reference_res.cols;
  for(size_t i = 0; i < total; ++i) {
    if(reference_res.es[i] != test_res.es[i]) {
      fputs("Matrices did not match", stderr);
      return;
    }
  }
  puts("Matrices are equal");
}