// gym.h is a UI framework built on top of nn.h and raylib.
// It provides reusable widgets for monitoring the training of models.
// Because logging in terminal is boring.

#ifndef GYM_H_
#define GYM_H_

#include <float.h>
#include <raylib.h>
#include <raymath.h>

#include "nn.h"

#ifndef GYM_ASSERT
#define GYM_ASSERT NN_ASSERT
#endif // GYM_ASSERT

// The Tsoding Background Color
#define GYM_BACKGROUND CLITERAL(Color) { 0x18, 0x18, 0x18, 0xFF }

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
#define gym_layout_stack_slot(ls) (GYM_ASSERT((ls)->count > 0), gym_layout_slot_loc(&(ls)->items[(ls)->count - 1], __FILE__, __LINE__))
#define gym_layout_stack_pop(ls) do { GYM_ASSERT((ls)->count > 0); (ls)->count -= 1; } while (0)

static Gym_Layout_Stack default_gym_layout_stack = {0};

#define gym_layout_begin(orient, rect, count, gap) gym_layout_stack_push(&default_gym_layout_stack, orient, rect, count, gap)
#define gym_layout_end() gym_layout_stack_pop(&default_gym_layout_stack)
// TODO: allow a single slot to take up several slots
#define gym_layout_slot() gym_layout_stack_slot(&default_gym_layout_stack)

#define DA_INIT_CAP 256
#define da_append(da, item)                                                          \
    do {                                                                             \
        if ((da)->count >= (da)->capacity) {                                         \
            (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity*2;   \
            (da)->items = realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
            GYM_ASSERT((da)->items != NULL && "Buy more RAM lol");                   \
        }                                                                            \
                                                                                     \
        (da)->items[(da)->count++] = (item);                                         \
    } while (0)

void gym_render_nn(NN nn, Gym_Rect r);
void gym_render_mat_as_heatmap(Mat m, Gym_Rect r, size_t max_width);
void gym_render_nn_weights_heatmap(NN nn, Gym_Rect r);
void gym_render_nn_activations_heatmap(NN nn, Gym_Rect r);
void gym_plot(Gym_Plot plot, Gym_Rect r);
void gym_slider(float *value, bool *dragging, float rx, float ry, float rw, float rh);
void gym_nn_image_grayscale(NN nn, void *pixels, size_t width, size_t height, size_t stride, float low, float high);

#endif // GYM_H_

#ifdef GYM_IMPLEMENTATION

void gym_render_nn(NN nn, Gym_Rect r)
{
    Color low_color = RED;
    Color high_color = DARKBLUE;

    float neuron_radius = r.h*0.03;
    float layer_border_vpad = r.h*0.08;
    float layer_border_hpad = r.w*0.06;
    float nn_width = r.w - 2*layer_border_hpad;
    float nn_height = r.h - 2*layer_border_vpad;
    float nn_x = r.x + r.w/2 - nn_width/2;
    float nn_y = r.y + r.h/2 - nn_height/2;
    float layer_hpad = nn_width / nn.arch_count;
    for (size_t l = 0; l < nn.arch_count; ++l) {
        float layer_vpad1 = nn_height / nn.as[l].cols;
        for (size_t i = 0; i < nn.as[l].cols; ++i) {
            float cx1 = nn_x + l*layer_hpad + layer_hpad/2;
            float cy1 = nn_y + i*layer_vpad1 + layer_vpad1/2;
            if (l+1 < nn.arch_count) {
                float layer_vpad2 = nn_height / nn.as[l+1].cols;
                for (size_t j = 0; j < nn.as[l+1].cols; ++j) {
                    // i - rows of ws
                    // j - cols of ws
                    float cx2 = nn_x + (l+1)*layer_hpad + layer_hpad/2;
                    float cy2 = nn_y + j*layer_vpad2 + layer_vpad2/2;
                    float value = sigmoidf(MAT_AT(nn.ws[l], i, j));
                    high_color.a = floorf(255.f*value);
                    float thick = r.h*0.004f;
                    Vector2 start = {cx1, cy1};
                    Vector2 end   = {cx2, cy2};
                    DrawLineEx(start, end, thick, ColorAlphaBlend(low_color, high_color, WHITE));
                }
            }
            if (l > 0) {
                high_color.a = floorf(255.f*sigmoidf(ROW_AT(nn.bs[l-1], i)));
                DrawCircle(cx1, cy1, neuron_radius, ColorAlphaBlend(low_color, high_color, WHITE));
            } else {
                DrawCircle(cx1, cy1, neuron_radius, GRAY);
            }
        }
    }
}

void gym_render_mat_as_heatmap(Mat m, Gym_Rect r, size_t max_width)
{
    Color low_color = RED;
    Color high_color = DARKBLUE;

    float cell_width = r.w*m.cols/max_width/m.cols;
    float cell_height = r.h/m.rows;

    float full_width = r.w*m.cols/max_width;

    for (size_t y = 0; y < m.rows; ++y) {
        for (size_t x = 0; x < m.cols; ++x) {
            high_color.a = floorf(255.f*sigmoidf(MAT_AT(m, y, x)));
            Color color = ColorAlphaBlend(low_color, high_color, WHITE);
            Gym_Rect slot = {
                r.x + r.w/2 - full_width/2 + x*cell_width,
                r.y + y*cell_height,
                cell_width,
                cell_height,
            };
            DrawRectangle(ceilf(slot.x), ceilf(slot.y), ceilf(slot.w), ceilf(slot.h), color);
        }
    }
}

void gym_render_nn_weights_heatmap(NN nn, Gym_Rect r)
{
    size_t max_width = 0;
    for (size_t i = 0; i < nn.arch_count - 1; ++i) {
        if (max_width < nn.ws[i].cols) {
            max_width = nn.ws[i].cols;
        }
    }

    gym_layout_begin(GLO_VERT, r, nn.arch_count - 1, 20);
    for (size_t i = 0; i < nn.arch_count - 1; ++i) {
        gym_render_mat_as_heatmap(nn.ws[i], gym_layout_slot(), max_width);
    }
    gym_layout_end();
}

void gym_render_nn_activations_heatmap(NN nn, Gym_Rect r)
{
    size_t max_width = 0;
    for (size_t i = 0; i < nn.arch_count; ++i) {
        if (max_width < nn.as[i].cols) {
            max_width = nn.as[i].cols;
        }
    }

    gym_layout_begin(GLO_VERT, r, nn.arch_count, 20);
    for (size_t i = 0; i < nn.arch_count; ++i) {
        gym_render_mat_as_heatmap(row_as_mat(nn.as[i]), gym_layout_slot(), max_width);
    }
    gym_layout_end();
}

void gym_plot(Gym_Plot plot, Gym_Rect r)
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
        float x1 = r.x + r.w/n*i;
        float y1 = r.y + (1 - (plot.items[i] - min)/(max - min))*r.h;
        float x2 = r.x + (float)r.w/n*(i+1);
        float y2 = r.y + (1 - (plot.items[i+1] - min)/(max - min))*r.h;
        DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, r.h*0.005, RED);
    }

    float y0 = r.y + (1 - (0 - min)/(max - min))*r.h;
    DrawLineEx((Vector2){r.x + 0, y0}, (Vector2){r.x + r.w - 1, y0}, r.h*0.005, WHITE);
    DrawText("0", r.x + 0, y0 - r.h*0.04, r.h*0.04, WHITE);
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
    GYM_ASSERT(NN_INPUT(nn).cols >= 2);
    GYM_ASSERT(NN_OUTPUT(nn).cols >= 1);
    uint32_t *pixels_u32 = pixels;
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            ROW_AT(NN_INPUT(nn), 0) = (float)x/(float)(width - 1);
            ROW_AT(NN_INPUT(nn), 1) = (float)y/(float)(height - 1);
            nn_forward(nn);
            float a = ROW_AT(NN_OUTPUT(nn), 0);
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
        r.w = (l->rect.w - l->gap*(l->count - 1))/l->count;
        r.h = l->rect.h;
        r.x = l->rect.x + l->i*(r.w + l->gap);
        r.y = l->rect.y;
        break;

    case GLO_VERT:
        r.w = l->rect.w;
        r.h = (l->rect.h - l->gap*(l->count - 1))/l->count;
        r.x = l->rect.x;
        r.y = l->rect.y + l->i*(r.h + l->gap);
        break;

    default:
        GYM_ASSERT(0 && "Unreachable");
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

#endif // GYM_IMPLEMENTATION
