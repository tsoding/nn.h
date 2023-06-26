#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

// #define NN_BACKPROP_TRADITIONAL

#ifndef NN_ACT
#define NN_ACT ACT_SIG
#endif // NN_ACT

#ifndef NN_RELU_PARAM
#define NN_RELU_PARAM 0.01f
#endif // NN_RELU_PARAM

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])

typedef enum {
    ACT_SIG,
    ACT_RELU,
    ACT_TANH,
    ACT_SIN,
} Act;

float rand_float(void);

float sigmoidf(float x);
float reluf(float x);
float tanhf(float x);

// Dispatch to the corresponding activation function
float actf(float x, Act act);

// Derivative of the activation function based on its value
float dactf(float y, Act act);

typedef struct {
    size_t capacity;
    size_t size;
    uintptr_t *words;
} Region;

// capacity is in bytes, but it can allocate more just to keep things
// word aligned
Region region_alloc_alloc(size_t capacity_bytes);
void *region_alloc(Region *r, size_t size_bytes);
#define region_reset(r) (NN_ASSERT((r) != NULL), (r)->size = 0)
#define region_occupied_bytes(r) (NN_ASSERT((r) != NULL), (r)->size*sizeof(*(r)->words))

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

Mat mat_alloc(Region *r, size_t rows, size_t cols);
void mat_save(FILE *out, Mat m);
Mat mat_load(FILE *in, Region *r);
void mat_fill(Mat m, float x);
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_act(Mat m);
void mat_print(Mat m, const char *name, size_t padding);
void mat_shuffle_rows(Mat m);
#define MAT_PRINT(m) mat_print(m, #m, 0)

typedef struct {
    size_t *arch;
    size_t arch_count;
    Mat *ws; // The amount of activations is arch_count-1
    Mat *bs; // The amount of activations is arch_count-1

    // TODO: maybe remove these? It would be better to allocate them in a
    // temporary region during the actual forwarding
    Mat *as;
} NN;

#define NN_INPUT(nn) (NN_ASSERT((nn).arch_count > 0), (nn).as[0])
#define NN_OUTPUT(nn) (NN_ASSERT((nn).arch_count > 0), (nn).as[(nn).arch_count-1])

NN nn_alloc(Region *r, size_t *arch, size_t arch_count);
void nn_zero(NN nn);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
NN nn_finite_diff(Region *r, NN nn, Mat ti, Mat to, float eps);
NN nn_backprop(Region *r, NN nn, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);

typedef struct {
    size_t begin;
    float cost;
    bool finished;
} Batch;

void batch_process(Region *r, Batch *b, size_t batch_size, NN nn, Mat t, float rate);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float reluf(float x)
{
    return x > 0 ? x : x*NN_RELU_PARAM;
}

float tanhf(float x)
{
    float ex = expf(x);
    float enx = expf(-x);
    return (ex - enx)/(ex + enx);
}

float actf(float x, Act act)
{
    switch (act) {
    case ACT_SIG:  return sigmoidf(x);
    case ACT_RELU: return reluf(x);
    case ACT_TANH: return tanhf(x);
    case ACT_SIN:  return sinf(x);
    }
    NN_ASSERT(0 && "Unreachable");
    return 0.0f;
}

float dactf(float y, Act act)
{
    switch (act) {
    case ACT_SIG:  return y*(1 - y);
    case ACT_RELU: return y >= 0 ? 1 : NN_RELU_PARAM;
    case ACT_TANH: return 1 - y*y;
    case ACT_SIN:  return cosf(asinf(y));
    }
    NN_ASSERT(0 && "Unreachable");
    return 0.0f;
}

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}

Mat mat_alloc(Region *r, size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = region_alloc(r, sizeof(*m.es)*rows*cols);
    NN_ASSERT(m.es != NULL);
    return m;
}

void mat_save(FILE *out, Mat m)
{
    const char *magic = "nn.h.mat";
    fwrite(magic, strlen(magic), 1, out);
    fwrite(&m.rows, sizeof(m.rows), 1, out);
    fwrite(&m.cols, sizeof(m.cols), 1, out);
    for (size_t i = 0; i < m.rows; ++i) {
        size_t n = fwrite(&MAT_AT(m, i, 0), sizeof(*m.es), m.cols, out);
        while (n < m.cols && !ferror(out)) {
            size_t k = fwrite(m.es + n, sizeof(*m.es), m.cols - n, out);
            n += k;
        }
    }
}

Mat mat_load(FILE *in, Region *r)
{
    uint64_t magic;
    fread(&magic, sizeof(magic), 1, in);
    NN_ASSERT(magic == 0x74616d2e682e6e6e);
    size_t rows, cols;
    fread(&rows, sizeof(rows), 1, in);
    fread(&cols, sizeof(cols), 1, in);
    Mat m = mat_alloc(r, rows, cols);

    size_t n = fread(m.es, sizeof(*m.es), rows*cols, in);
    while (n < rows*cols && !ferror(in)) {
        size_t k = fread(m.es, sizeof(*m.es) + n, rows*cols - n, in);
        n += k;
    }

    return m;
}

void mat_dot(Mat dst, Mat a, Mat b)
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

Mat mat_row(Mat m, size_t row)
{
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0),
    };
}

void mat_copy(Mat dst, Mat src)
{
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_sum(Mat dst, Mat a)
{
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_act(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = actf(MAT_AT(m, i, j), NN_ACT);
        }
    }
}

void mat_print(Mat m, const char *name, size_t padding)
{
    printf("%*s%s = [\n", (int) padding, "", name);
    for (size_t i = 0; i < m.rows; ++i) {
        printf("%*s    ", (int) padding, "");
        for (size_t j = 0; j < m.cols; ++j) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}

void mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = x;
        }
    }
}

void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = rand_float()*(high - low) + low;
        }
    }
}

NN nn_alloc(Region *r, size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);

    NN nn;
    nn.arch = arch;
    nn.arch_count = arch_count;

    nn.ws = region_alloc(r, sizeof(*nn.ws)*(nn.arch_count - 1));
    NN_ASSERT(nn.ws != NULL);
    nn.bs = region_alloc(r, sizeof(*nn.bs)*(nn.arch_count - 1));
    NN_ASSERT(nn.bs != NULL);
    nn.as = region_alloc(r, sizeof(*nn.as)*nn.arch_count);
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(r, 1, arch[0]);
    for (size_t i = 1; i < arch_count; ++i) {
        nn.ws[i-1] = mat_alloc(r, nn.as[i-1].cols, arch[i]);
        nn.bs[i-1] = mat_alloc(r, 1, arch[i]);
        nn.as[i]   = mat_alloc(r, 1, arch[i]);
    }

    return nn;
}

void nn_zero(NN nn)
{
    for (size_t i = 0; i < nn.arch_count - 1; ++i) {
        mat_fill(nn.ws[i], 0);
        mat_fill(nn.bs[i], 0);
        mat_fill(nn.as[i], 0);
    }
    mat_fill(nn.as[nn.arch_count - 1], 0);
}

void nn_print(NN nn, const char *name)
{
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.arch_count-1; ++i) {
        snprintf(buf, sizeof(buf), "ws%zu", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        mat_print(nn.bs[i], buf, 4);
    }
    printf("]\n");
}

void nn_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.arch_count-1; ++i) {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.arch_count-1; ++i) {
        mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i+1], nn.bs[i]);
        mat_act(nn.as[i+1]);
    }
}

float nn_cost(NN nn, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);
    size_t n = ti.rows;

    float c = 0;
    for (size_t i = 0; i < n; ++i) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);
        size_t q = to.cols;
        for (size_t j = 0; j < q; ++j) {
            float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            c += d*d;
        }
    }

    return c/n;
}

NN nn_backprop(Region *r, NN nn, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    size_t n = ti.rows;
    NN_ASSERT(NN_OUTPUT(nn).cols == to.cols);

    NN g = nn_alloc(r, nn.arch, nn.arch_count);
    nn_zero(g);

    // i - current sample
    // l - current layer
    // j - current activation
    // k - previous activation

    for (size_t i = 0; i < n; ++i) {
        mat_copy(NN_INPUT(nn), mat_row(ti, i));
        nn_forward(nn);

        for (size_t j = 0; j < nn.arch_count; ++j) {
            mat_fill(g.as[j], 0);
        }

        for (size_t j = 0; j < to.cols; ++j) {
#ifdef NN_BACKPROP_TRADITIONAL
            MAT_AT(NN_OUTPUT(g), 0, j) = 2*(MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j));
#else
            MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
#endif // NN_BACKPROP_TRADITIONAL
        }

#ifdef NN_BACKPROP_TRADITIONAL
        float s = 1;
#else
        float s = 2;
#endif // NN_BACKPROP_TRADITIONAL

        for (size_t l = nn.arch_count-1; l > 0; --l) {
            for (size_t j = 0; j < nn.as[l].cols; ++j) {
                float a = MAT_AT(nn.as[l], 0, j);
                float da = MAT_AT(g.as[l], 0, j);
                float qa = dactf(a, NN_ACT);
                MAT_AT(g.bs[l-1], 0, j) += s*da*qa;
                for (size_t k = 0; k < nn.as[l-1].cols; ++k) {
                    // j - weight matrix col
                    // k - weight matrix row
                    float pa = MAT_AT(nn.as[l-1], 0, k);
                    float w = MAT_AT(nn.ws[l-1], k, j);
                    MAT_AT(g.ws[l-1], k, j) += s*da*qa*pa;
                    MAT_AT(g.as[l-1], 0, k) += s*da*qa*w;
                }
            }
        }
    }

    for (size_t i = 0; i < g.arch_count-1; ++i) {
        for (size_t j = 0; j < g.ws[i].rows; ++j) {
            for (size_t k = 0; k < g.ws[i].cols; ++k) {
                MAT_AT(g.ws[i], j, k) /= n;
            }
        }
        for (size_t j = 0; j < g.bs[i].rows; ++j) {
            for (size_t k = 0; k < g.bs[i].cols; ++k) {
                MAT_AT(g.bs[i], j, k) /= n;
            }
        }
    }

    return g;
}

NN nn_finite_diff(Region *r, NN nn, Mat ti, Mat to, float eps)
{
    float saved;
    float c = nn_cost(nn, ti, to);

    NN g = nn_alloc(r, nn.arch, nn.arch_count);

    for (size_t i = 0; i < nn.arch_count-1; ++i) {
        for (size_t j = 0; j < nn.ws[i].rows; ++j) {
            for (size_t k = 0; k < nn.ws[i].cols; ++k) {
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += eps;
                MAT_AT(g.ws[i], j, k) = (nn_cost(nn, ti, to) - c)/eps;
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; ++j) {
            for (size_t k = 0; k < nn.bs[i].cols; ++k) {
                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) += eps;
                MAT_AT(g.bs[i], j, k) = (nn_cost(nn, ti, to) - c)/eps;
                MAT_AT(nn.bs[i], j, k) = saved;
            }
        }
    }

    return g;
}

void nn_learn(NN nn, NN g, float rate)
{
    for (size_t i = 0; i < nn.arch_count-1; ++i) {
        for (size_t j = 0; j < nn.ws[i].rows; ++j) {
            for (size_t k = 0; k < nn.ws[i].cols; ++k) {
                MAT_AT(nn.ws[i], j, k) -= rate*MAT_AT(g.ws[i], j, k);
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; ++j) {
            for (size_t k = 0; k < nn.bs[i].cols; ++k) {
                MAT_AT(nn.bs[i], j, k) -= rate*MAT_AT(g.bs[i], j, k);
            }
        }
    }
}

void mat_shuffle_rows(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
         size_t j = i + rand()%(m.rows - i);
         if (i != j) {
             for (size_t k = 0; k < m.cols; ++k) {
                 float t = MAT_AT(m, i, k);
                 MAT_AT(m, i, k) = MAT_AT(m, j, k);
                 MAT_AT(m, j, k) = t;
             }
         }
    }
}

void batch_process(Region *r, Batch *b, size_t batch_size, NN nn, Mat t, float rate)
{
    if (b->finished) {
        b->finished = false;
        b->begin = 0;
        b->cost = 0;
    }

    size_t size = batch_size;
    if (b->begin + batch_size >= t.rows)  {
        size = t.rows - b->begin;
    }

    Mat batch_ti = {
        .rows = size,
        .cols = NN_INPUT(nn).cols,
        .stride = t.stride,
        .es = &MAT_AT(t, b->begin, 0),
    };

    Mat batch_to = {
        .rows = size,
        .cols = NN_OUTPUT(nn).cols,
        .stride = t.stride,
        .es = &MAT_AT(t, b->begin, batch_ti.cols),
    };

    NN g = nn_backprop(r, nn, batch_ti, batch_to);
    nn_learn(nn, g, rate);
    b->cost += nn_cost(nn, batch_ti, batch_to);
    b->begin += batch_size;

    if (b->begin >= t.rows) {
        size_t batch_count = (t.rows + batch_size - 1)/batch_size;
        b->cost /= batch_count;
        b->finished = true;
    }
}

Region region_alloc_alloc(size_t capacity_bytes)
{
    Region r = {0};

    size_t word_size = sizeof(*r.words);
    size_t capacity_words = (capacity_bytes + word_size - 1)/word_size;

    void *words = NN_MALLOC(capacity_words*word_size);
    NN_ASSERT(words != NULL);
    r.capacity = capacity_words;
    r.words = words;
    return r;
}

void *region_alloc(Region *r, size_t size_bytes)
{
    if (r == NULL) return NN_MALLOC(size_bytes);
    size_t word_size = sizeof(*r->words);
    size_t size_words = (size_bytes + word_size - 1)/word_size;

    NN_ASSERT(r->size + size_words <= r->capacity);
    if (r->size + size_words > r->capacity) return NULL;
    void *result = &r->words[r->size];
    r->size += size_words;
    return result;
}

#endif // NN_IMPLEMENTATION
