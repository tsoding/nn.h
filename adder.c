#define NN_IMPLEMENTATION
#define NN_ENABLE_GYM
#include "nn.h"

#define BITS 5

size_t arch[] = {2*BITS, 4*BITS, BITS + 1};
size_t epoch = 0;
size_t max_epoch = 100*1000;
size_t batches_per_frame = 200;
size_t batch_size = 28;
float rate = 1.0f;
bool paused = true;

void verify_nn_adder(Font font, NN nn, Gym_Rect r)
{
    float s;
    if (r.w < r.h) {
        s = r.w - r.w*0.05;
        r.y = r.y + r.h/2 - s/2;
    } else {
        s = r.h - r.h*0.05;
        r.x = r.x + r.w/2 - s/2;
    }
    size_t n = 1<<BITS;
    float cs = s/n;

    for (size_t x = 0; x < n; ++x) {
        for (size_t y = 0; y < n; ++y) {
            for (size_t i = 0; i < BITS; ++i) {
                MAT_AT(NN_INPUT(nn), 0, i)        = (x>>i)&1;
                MAT_AT(NN_INPUT(nn), 0, i + BITS) = (y>>i)&1;
            }

            nn_forward(nn);

            size_t z = 0.0f;
            for (size_t i = 0; i < BITS; ++i) {
                size_t bit = MAT_AT(NN_OUTPUT(nn), 0, i) > 0.5;
                z = z|(bit<<i);
            }
            bool overflow = MAT_AT(NN_OUTPUT(nn), 0, BITS) > 0.5;
            bool correct = z == x + y;

            Vector2 position = { r.x + x*cs, r.y + y*cs };
            Vector2 size = { cs, cs };

            if (correct)  DrawRectangleV(position, size, DARKGREEN);
            if (overflow) DrawRectangleV(position, size, DARKPURPLE);

            char buffer[256];
            snprintf(buffer, sizeof(buffer), "%zu", z);

            // Centering the text
            float fontSize = cs*0.8;
            float spacing = 0;
            Vector2 text_size = MeasureTextEx(font, buffer, fontSize, spacing);
            position.x = position.x + cs/2 - text_size.x/2;
            position.y = position.y + cs/2 - text_size.y/2;

            DrawTextEx(font, buffer, position, fontSize, spacing, WHITE);
        }
    }
}

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
        if (z >= n) {
            for (size_t j = 0; j < BITS; ++j) {
                MAT_AT(to, i, j) = 1;
            }
            MAT_AT(to, i, BITS) = 1;
        } else {
            MAT_AT(to, i, BITS) = 0;
        }
    }

    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);

    size_t WINDOW_FACTOR = 80;
    size_t WINDOW_WIDTH = (16*WINDOW_FACTOR);
    size_t WINDOW_HEIGHT = (9*WINDOW_FACTOR);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "xor");
    SetTargetFPS(60);

    Font font = LoadFontEx("./fonts/iosevka-regular.ttf", 72, NULL, 0);
    SetTextureFilter(font.texture, TEXTURE_FILTER_BILINEAR);

    Gym_Plot plot = {0};
    Batch batch = {0};

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_R)) {
            epoch = 0;
            nn_rand(nn, -1, 1);
            plot.count = 0;
        }

        for (size_t i = 0; i < batches_per_frame && !paused && epoch < max_epoch; ++i) {
            batch_process(&batch, batch_size, nn, g, t, rate);
            if (batch.finished) {
                epoch += 1;
                da_append(&plot, batch.cost);
                mat_shuffle_rows(t);
            }
        }

        BeginDrawing();
        Color background_color = {0x18, 0x18, 0x18, 0xFF};
        ClearBackground(background_color);
        {
            int w = GetRenderWidth();
            int h = GetRenderHeight();

            Gym_Rect r;
            r.w = w;
            r.h = h*2/3;
            r.x = 0;
            r.y = h/2 - r.h/2;

            gym_layout_begin(GLO_HORZ, r, 3, 10);
                gym_plot(plot, gym_layout_slot());
                gym_layout_begin(GLO_VERT, gym_layout_slot(), 2, 0);
                    gym_render_nn(nn, gym_layout_slot());
                    gym_render_nn_weights_heatmap(nn, gym_layout_slot());
                gym_layout_end();
                verify_nn_adder(font, nn, gym_layout_slot());
            gym_layout_end();

            char buffer[256];
            snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu, Rate: %f, Cost: %f", epoch, max_epoch, rate, nn_cost(nn, ti, to));
            DrawTextEx(font, buffer, CLITERAL(Vector2){}, h*0.04, 0, WHITE);
        }
        EndDrawing();
    }


    return 0;
}
