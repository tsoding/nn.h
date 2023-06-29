#include <stdio.h>
#include <time.h>

#define OLIVEC_AA_RES 1
#define OLIVEC_IMPLEMENTATION
#include "olive.c"

#define NN_BACKPROP_TRADITIONAL
#define NN_ACT ACT_SIG

#define GYM_IMPLEMENTATION
#include "gym.h"

#define NN_IMPLEMENTATION
#include "nn.h"

#define WIDTH 28
#define HEIGHT WIDTH
enum {
    SHAPE_CIRCLE,
    SHAPE_RECT,
    SHAPES,
};
#define TRAINING_SAMPLES_PER_SHAPE 2000
#define VERIFICATION_SAMPLES_PER_SHAPE (TRAINING_SAMPLES_PER_SHAPE/2)
#define BACKGROUND_COLOR 0xFF000000
#define FOREGROUND_COLOR 0xFFFFFFFF

size_t arch[] = {WIDTH*HEIGHT, 14, 7, 5, SHAPES};
size_t batch_size = 20;
size_t batches_per_frame = 20;
float rate = 0.1f;
bool paused = true;

void random_boundary(size_t width, size_t height, int *x1, int *y1, int *w, int *h)
{
    int x2, y2, i = 0;
    do {
        *x1 = rand()%width;
        *y1 = rand()%height;
        x2 = rand()%width;
        y2 = rand()%height;
        if (*x1 > x2) OLIVEC_SWAP(int, *x1, x2);
        if (*y1 > y2) OLIVEC_SWAP(int, *y1, y2);
        *w = x2 - *x1;
        *h = y2 - *y1;
    } while ((*w < 4 || *h < 4) && i++ < 100);
    assert(*w >= 4 && *h >= 4);
}

void random_circle(Olivec_Canvas oc)
{
    int x, y, w, h;
    random_boundary(oc.width, oc.height, &x, &y, &w, &h);
    olivec_fill(oc, BACKGROUND_COLOR);
    int r = (w < h ? w : h)/2;
    olivec_circle(oc, x + w/2, y + h/2, r, FOREGROUND_COLOR);
}

void random_rect(Olivec_Canvas oc)
{
    int x, y, w, h;
    random_boundary(oc.width, oc.height, &x, &y, &w, &h);
    olivec_fill(oc, BACKGROUND_COLOR);
    olivec_rect(oc, x, y, w, h, FOREGROUND_COLOR);
}

void canvas_to_row(Olivec_Canvas oc, Row row)
{
    NN_ASSERT(oc.width*oc.height == row.cols);
    for (size_t y = 0; y < oc.height; ++y){
        for (size_t x = 0; x < oc.width; ++x) {
            ROW_AT(row, y*oc.width + x) = (float)(OLIVEC_PIXEL(oc, x, y)&0xFF)/255.f;
        }
    }
}

Mat generate_samples(Region *r, size_t samples)
{
    size_t input_size = WIDTH*HEIGHT;
    size_t output_size = SHAPES;
    Mat t = mat_alloc(r, samples*SHAPES, input_size + output_size);
    size_t s = region_save(r);
        Olivec_Canvas oc = {0};
        oc.pixels = region_alloc(r, WIDTH*HEIGHT*sizeof(*oc.pixels));
        oc.width = WIDTH;
        oc.height = HEIGHT;
        oc.stride = WIDTH;
        for (size_t i = 0; i < samples; ++i) {
            int x, y, w, h;
            random_boundary(oc.width, oc.height, &x, &y, &w, &h);
            int r = (w < h ? w : h)/2;
            for (size_t j = 0; j < SHAPES; ++j) {
                Row row = mat_row(t, i*2 + j);
                Row in = row_slice(row, 0, input_size);
                Row out = row_slice(row, input_size, output_size);
                olivec_fill(oc, BACKGROUND_COLOR);
                switch (j) {
                case SHAPE_CIRCLE: olivec_circle(oc, x + w/2, y + h/2, r, FOREGROUND_COLOR); break;
                case SHAPE_RECT:   olivec_rect(oc, x, y, w, h, FOREGROUND_COLOR);  break;
                default: assert(0 && "unreachable");
                }
                canvas_to_row(oc, in);
                row_fill(out, 0);
                ROW_AT(out, j) = 1.0f;
            }
        }
    region_rewind(r, s);
    return t;
}

Gym_Rect gym_fit_square(Gym_Rect r)
{
    if (r.w < r.h) {
        return (Gym_Rect) {
            .x = r.x,
            .y = r.y + r.h/2 - r.w/2,
            .w = r.w,
            .h = r.w,
        };
    } else {
        return (Gym_Rect) {
            .x = r.x + r.w/2 - r.h/2,
            .y = r.y,
            .w = r.h,
            .h = r.h,
        };
    }
}

void gym_drawable_canvas(Olivec_Canvas oc, Gym_Rect r)
{
    NN_ASSERT(oc.width == oc.height && "We support only square canvases");
    r = gym_fit_square(r);
    float cw = r.w/oc.width;
    float ch = r.h/oc.height;

    Vector2 mouse = GetMousePosition();
    Rectangle boundary = { r.x, r.y, r.w, r.h };
    if (CheckCollisionPointRec(mouse, boundary)) {
        size_t x = (mouse.x - r.x)/cw;
        size_t y = (mouse.y - r.y)/ch;
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            OLIVEC_PIXEL(oc, x, y) = FOREGROUND_COLOR;
        }
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            OLIVEC_PIXEL(oc, x, y) = BACKGROUND_COLOR;
        }
    }

    for (size_t y = 0; y < oc.height; ++y) {
        for (size_t x = 0; x < oc.width; ++x) {
            DrawRectangle(
                ceilf(r.x + x*cw),
                ceilf(r.y + y*ch),
                ceilf(cw),
                ceilf(ch),
                *(Color*)&OLIVEC_PIXEL(oc, x, y));
        }
    }
}

Gym_Rect gym_root(void)
{
    Gym_Rect root = {0};
    root.w = GetRenderWidth();
    root.h = GetRenderHeight();
    return root;
}

void display_training_data(Mat t)
{
    for (size_t i = 0; i < t.rows; ++i) {
        Row row = mat_row(t, i);
        Row in = row_slice(row, 0, WIDTH*HEIGHT);
        Row out = row_slice(row, WIDTH*HEIGHT, SHAPES);
        for (size_t y = 0; y < HEIGHT; ++y) {
            for (size_t x = 0; x < WIDTH; ++x) {
                if (ROW_AT(in, y*WIDTH + x) > 1e-6f) {
                    printf("##");
                } else {
                    printf("  ");
                }
            }
            printf("\n");
        }
        MAT_PRINT(row_as_mat(out));
    }
}

int main(void)
{
    srand(time(0));

    Region temp = region_alloc_alloc(256*1024*1024);
    Region main = region_alloc_alloc(256*1024*1024);

    NN nn = nn_alloc(&main, arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);
    Mat t = generate_samples(&main, TRAINING_SAMPLES_PER_SHAPE);
    Mat v = generate_samples(&main, VERIFICATION_SAMPLES_PER_SHAPE);

    Gym_Plot tplot = {0};
    Gym_Plot vplot = {0};
    Batch batch = {0};

    int factor = 80;
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(16*factor, 9*factor, "Shape");
    SetTargetFPS(60);

    Olivec_Canvas canvas = {0};
    canvas.pixels = region_alloc(&main, WIDTH*HEIGHT*sizeof(*canvas.pixels));
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    canvas.stride = WIDTH;
    olivec_fill(canvas, BACKGROUND_COLOR);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_R)) {
            nn_rand(nn, -1, 1);
            tplot.count = 0;
            vplot.count = 0;
        }
        if (IsKeyPressed(KEY_C)) {
            olivec_fill(canvas, BACKGROUND_COLOR);
        }
        if (IsKeyPressed(KEY_Q)) {
            random_circle(canvas);
        }
        if (IsKeyPressed(KEY_W)) {
            random_rect(canvas);
        }

        for (size_t i = 0; i < batches_per_frame && !paused; ++i) {
            size_t s = region_save(&temp);
            batch_process(&temp, &batch, batch_size, nn, t, rate);
            if (batch.finished) {
                da_append(&tplot, batch.cost);
                mat_shuffle_rows(t);
                da_append(&vplot, nn_cost(nn, v));
            }
            region_rewind(&temp, s);
        }

        BeginDrawing();
            ClearBackground(GYM_BACKGROUND);
            gym_layout_begin(GLO_HORZ, gym_root(), 2, 10);
                gym_layout_begin(GLO_VERT, gym_layout_slot(), 2, 10);
                    gym_plot(tplot, gym_layout_slot(), RED);
                    gym_plot(vplot, gym_layout_slot(), GREEN);
                gym_layout_end();
                gym_layout_begin(GLO_VERT, gym_layout_slot(), 2, 10);
                    gym_drawable_canvas(canvas, gym_layout_slot());
                    canvas_to_row(canvas, NN_INPUT(nn));
                    nn_forward(nn);
                    {
                        Gym_Rect slot = gym_layout_slot();
                        gym_render_mat_as_heatmap(row_as_mat(NN_OUTPUT(nn)), slot, NN_OUTPUT(nn).cols);
                        if (ROW_AT(NN_OUTPUT(nn), 0) > ROW_AT(NN_OUTPUT(nn), 1)) {
                            DrawText("circle", slot.x, slot.y, slot.h*0.08, WHITE);
                        } else if (ROW_AT(NN_OUTPUT(nn), 0) < ROW_AT(NN_OUTPUT(nn), 1)) {
                            DrawText("rectangle", slot.x, slot.y, slot.h*0.08, WHITE);
                        }
                    }
                gym_layout_end();
            gym_layout_end();
        EndDrawing();
    }

    CloseWindow();

    return 0;
}
