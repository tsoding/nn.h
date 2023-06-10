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
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

Mat mat_alloc(size_t rows, size_t cols);
void mat_save(FILE *out, Mat m);
Mat mat_load(FILE *in);
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
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; // The amount of activations is count+1
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_zero(NN nn);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to);
void nn_backprop(NN nn, NN g, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);

typedef struct {
    size_t begin;
    float cost;
    bool finished;
} Batch;

void batch_process(Batch *b, size_t batch_size, NN nn, NN g, Mat t, float rate);

#ifdef NN_ENABLE_GYM
#include <float.h>
#include <raylib.h>
#include <raymath.h>

typedef struct {
    float x;
    float y;
    float w;
    float h;
} Gym_Rect;

Gym_Rect gym_rect(float x, float y, float w, float h);

typedef enum {
    GLO_HORZ,
    GLO_VERT,
} Gym_Layout_Orient;

typedef struct {
    float *items;
    size_t count;
    size_t capacity;
} Gym_Plot;

typedef struct {
    Gym_Layout_Orient orient;
    Gym_Rect rect;
    size_t count;
    size_t i;
    float gap;
} Gym_Layout;

Gym_Rect gym_layout_slot_loc(Gym_Layout *l, const char *file_path, int line);

typedef struct {
    Gym_Layout *items;
    size_t count;
    size_t capacity;
} Gym_Layout_Stack;

void gym_layout_stack_push(Gym_Layout_Stack *ls, Gym_Layout_Orient orient, Gym_Rect rect, size_t count, float gap);
#define gls_push gym_layout_stack_push
#define gym_layout_stack_slot(ls) (assert((ls)->count > 0), gym_layout_slot_loc(&(ls)->items[(ls)->count - 1], __FILE__, __LINE__))
#define gls_slot gym_layout_stack_slot
#define gym_layout_stack_pop(ls) do { assert((ls)->count > 0); (ls)->count -= 1; } while (0)
#define gls_pop gym_layout_stack_pop


#define DA_INIT_CAP 256
#define da_append(da, item)                                                          \
    do {                                                                             \
        if ((da)->count >= (da)->capacity) {                                         \
            (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity*2;   \
            (da)->items = realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
            assert((da)->items != NULL && "Buy more RAM lol");                       \
        }                                                                            \
                                                                                     \
        (da)->items[(da)->count++] = (item);                                         \
    } while (0)

void gym_render_nn(NN nn, float rx, float ry, float rw, float rh);
void gym_plot(Gym_Plot plot, int rx, int ry, int rw, int rh);
void gym_slider(float *value, bool *dragging, float rx, float ry, float rw, float rh);
void gym_nn_image_grayscale(NN nn, void *pixels, size_t width, size_t height, size_t stride, float low, float high);

#endif // NN_ENABLE_GYM

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

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(sizeof(*m.es)*rows*cols);
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

Mat mat_load(FILE *in)
{
    uint64_t magic;
    fread(&magic, sizeof(magic), 1, in);
    NN_ASSERT(magic == 0x74616d2e682e6e6e);
    size_t rows, cols;
    fread(&rows, sizeof(rows), 1, in);
    fread(&cols, sizeof(cols), 1, in);
    Mat m = mat_alloc(rows, cols);

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

NN nn_alloc(size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);

    NN nn;
    nn.count = arch_count - 1;

    nn.ws = NN_MALLOC(sizeof(*nn.ws)*nn.count);
    NN_ASSERT(nn.ws != NULL);
    nn.bs = NN_MALLOC(sizeof(*nn.bs)*nn.count);
    NN_ASSERT(nn.bs != NULL);
    nn.as = NN_MALLOC(sizeof(*nn.as)*(nn.count + 1));
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1, arch[0]);
    for (size_t i = 1; i < arch_count; ++i) {
        nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
        nn.bs[i-1] = mat_alloc(1, arch[i]);
        nn.as[i]   = mat_alloc(1, arch[i]);
    }

    return nn;
}

void nn_zero(NN nn)
{
    for (size_t i = 0; i < nn.count; ++i) {
        mat_fill(nn.ws[i], 0);
        mat_fill(nn.bs[i], 0);
        mat_fill(nn.as[i], 0);
    }
    mat_fill(nn.as[nn.count], 0);
}

void nn_print(NN nn, const char *name)
{
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count; ++i) {
        snprintf(buf, sizeof(buf), "ws%zu", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        mat_print(nn.bs[i], buf, 4);
    }
    printf("]\n");
}

void nn_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count; ++i) {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.count; ++i) {
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

void nn_backprop(NN nn, NN g, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    size_t n = ti.rows;
    NN_ASSERT(NN_OUTPUT(nn).cols == to.cols);

    nn_zero(g);

    // i - current sample
    // l - current layer
    // j - current activation
    // k - previous activation

    for (size_t i = 0; i < n; ++i) {
        mat_copy(NN_INPUT(nn), mat_row(ti, i));
        nn_forward(nn);

        for (size_t j = 0; j <= nn.count; ++j) {
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

        for (size_t l = nn.count; l > 0; --l) {
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

    for (size_t i = 0; i < g.count; ++i) {
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
}

void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to)
{
    float saved;
    float c = nn_cost(nn, ti, to);

    for (size_t i = 0; i < nn.count; ++i) {
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
}

void nn_learn(NN nn, NN g, float rate)
{
    for (size_t i = 0; i < nn.count; ++i) {
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

void batch_process(Batch *b, size_t batch_size, NN nn, NN g, Mat t, float rate)
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

    nn_backprop(nn, g, batch_ti, batch_to);
    nn_learn(nn, g, rate);
    b->cost += nn_cost(nn, batch_ti, batch_to);
    b->begin += batch_size;

    if (b->begin >= t.rows) {
        size_t batch_count = (t.rows + batch_size - 1)/batch_size;
        b->cost /= batch_count;
        b->finished = true;
    }
}

#ifdef NN_ENABLE_GYM

void gym_render_nn(NN nn, float rx, float ry, float rw, float rh)
{
    Color low_color        = {0xFF, 0x00, 0xFF, 0xFF};
    Color high_color       = {0x00, 0xFF, 0x00, 0xFF};

    float neuron_radius = rh*0.03;
    float layer_border_vpad = rh*0.08;
    float layer_border_hpad = rw*0.06;
    float nn_width = rw - 2*layer_border_hpad;
    float nn_height = rh - 2*layer_border_vpad;
    float nn_x = rx + rw/2 - nn_width/2;
    float nn_y = ry + rh/2 - nn_height/2;
    size_t arch_count = nn.count + 1;
    float layer_hpad = nn_width / arch_count;
    for (size_t l = 0; l < arch_count; ++l) {
        float layer_vpad1 = nn_height / nn.as[l].cols;
        for (size_t i = 0; i < nn.as[l].cols; ++i) {
            float cx1 = nn_x + l*layer_hpad + layer_hpad/2;
            float cy1 = nn_y + i*layer_vpad1 + layer_vpad1/2;
            if (l+1 < arch_count) {
                float layer_vpad2 = nn_height / nn.as[l+1].cols;
                for (size_t j = 0; j < nn.as[l+1].cols; ++j) {
                    // i - rows of ws
                    // j - cols of ws
                    float cx2 = nn_x + (l+1)*layer_hpad + layer_hpad/2;
                    float cy2 = nn_y + j*layer_vpad2 + layer_vpad2/2;
                    float value = sigmoidf(MAT_AT(nn.ws[l], i, j));
                    high_color.a = floorf(255.f*value);
                    float thick = rh*0.004f;
                    Vector2 start = {cx1, cy1};
                    Vector2 end   = {cx2, cy2};
                    DrawLineEx(start, end, thick, ColorAlphaBlend(low_color, high_color, WHITE));
                }
            }
            if (l > 0) {
                high_color.a = floorf(255.f*sigmoidf(MAT_AT(nn.bs[l-1], 0, i)));
                DrawCircle(cx1, cy1, neuron_radius, ColorAlphaBlend(low_color, high_color, WHITE));
            } else {
                DrawCircle(cx1, cy1, neuron_radius, GRAY);
            }
        }
    }
}

void gym_plot(Gym_Plot plot, int rx, int ry, int rw, int rh)
{
    float min = FLT_MAX, max = FLT_MIN;
    for (size_t i = 0; i < plot.count; ++i) {
        if (max < plot.items[i]) max = plot.items[i];
        if (min > plot.items[i]) min = plot.items[i];
    }

    if (min > 0) min = 0;
    size_t n = plot.count;
    if (n < 1000) n = 1000;
    for (size_t i = 0; i+1 < plot.count; ++i) {
        float x1 = rx + (float)rw/n*i;
        float y1 = ry + (1 - (plot.items[i] - min)/(max - min))*rh;
        float x2 = rx + (float)rw/n*(i+1);
        float y2 = ry + (1 - (plot.items[i+1] - min)/(max - min))*rh;
        DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, rh*0.005, RED);
    }

    float y0 = ry + (1 - (0 - min)/(max - min))*rh;
    DrawLineEx((Vector2){rx + 0, y0}, (Vector2){rx + rw - 1, y0}, rh*0.005, WHITE);
    DrawText("0", rx + 0, y0 - rh*0.04, rh*0.04, WHITE);
}

void gym_slider(float *value, bool *dragging, float rx, float ry, float rw, float rh)
{
    float knob_radius = rh;
    Vector2 bar_size = {
        .x = rw - 2*knob_radius,
        .y = rh*0.25,
    };
    Vector2 bar_position = {
        .x = rx + knob_radius,
        .y = ry + rh/2 - bar_size.y/2
    };
    DrawRectangleV(bar_position, bar_size, WHITE);

    Vector2 knob_position = {
        .x = bar_position.x + bar_size.x*(*value),
        .y = ry + rh/2
    };
    DrawCircleV(knob_position, knob_radius, RED);

    if (*dragging) {
        float x = GetMousePosition().x;
        if (x < bar_position.x) x = bar_position.x;
        if (x > bar_position.x + bar_size.x) x = bar_position.x + bar_size.x;
        *value = (x - bar_position.x)/bar_size.x;
    }

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse_position = GetMousePosition();
        if (Vector2Distance(mouse_position, knob_position) <= knob_radius) {
            *dragging = true;
        }
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        *dragging = false;
    }
}

void gym_nn_image_grayscale(NN nn, void *pixels, size_t width, size_t height, size_t stride, float low, float high)
{
    NN_ASSERT(NN_INPUT(nn).cols >= 2);
    NN_ASSERT(NN_OUTPUT(nn).cols >= 1);
    uint32_t *pixels_u32 = pixels;
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(float)(width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(float)(height - 1);
            nn_forward(nn);
            float a = MAT_AT(NN_OUTPUT(nn), 0, 0);
            if (a < low) a = low;
            if (a > high) a = high;
            uint32_t pixel = (a + low)/(high - low)*255.f;
            pixels_u32[y*stride + x] = (0xFF<<(8*3))|(pixel<<(8*2))|(pixel<<(8*1))|(pixel<<(8*0));
        }
    }
}

Gym_Rect gym_rect(float x, float y, float w, float h)
{
    Gym_Rect r = {0};
    r.x = x;
    r.y = y;
    r.w = w;
    r.h = h;
    return r;
}

Gym_Rect gym_layout_slot_loc(Gym_Layout *l, const char *file_path, int line)
{
    if (l->i >= l->count) {
        fprintf(stderr, "%s:%d: ERROR: Layout overflow\n", file_path, line);
        exit(1);
    }

    Gym_Rect r = {0};

    switch (l->orient) {
    case GLO_HORZ:
        r.w = (l->rect.w + l->gap)/l->count - l->gap;
        r.h = l->rect.h;
        r.x = l->rect.x + l->i*(r.w + l->gap);
        r.y = l->rect.y;

        break;

    case GLO_VERT:
        r.w = l->rect.w;
        r.h = (l->rect.h + l->gap)/l->count - l->gap;
        r.x = l->rect.x;
        r.y = l->rect.y + l->i*(r.h + l->gap);

        break;

    default:
        assert(0 && "Unreachable");
    }

    l->i += 1;

    return r;
}

void gym_layout_stack_push(Gym_Layout_Stack *ls, Gym_Layout_Orient orient, Gym_Rect rect, size_t count, float gap)
{
    Gym_Layout l = {0};
    l.orient = orient;
    l.rect = rect;
    l.count = count;
    l.gap = gap;
    da_append(ls, l);
}

#endif // NN_ENABLE_GYM

#endif // NN_IMPLEMENTATION
