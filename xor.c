#define NN_IMPLEMENTATION
#define NN_ENABLE_GYM
#include "nn.h"

size_t arch[] = {2, 2, 1};
size_t max_epoch = 100*1000;
size_t epochs_per_frame = 103;
float rate = 1.0f;
bool paused = true;

void verify_nn_gate(Font font, NN nn, float rx, float ry, float rw, float rh)
{
    (void) rw;
    char buffer[256];
    float s = rh*0.06;
    float pad = rh*0.03;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = j;
            nn_forward(nn);
            snprintf(buffer, sizeof(buffer), "%zu @ %zu == %f", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
            DrawTextEx(font, buffer, CLITERAL(Vector2){rx, ry + (i*2 + j)*(s + pad)}, s, 0, WHITE);
        }
    }
}

int main(void)
{
    Mat t = mat_alloc(4, 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            size_t row = i*2 + j;
            MAT_AT(t, row, 0) = i;
            MAT_AT(t, row, 1) = j;
            MAT_AT(t, row, 2) = i^j;
        }
    }

    Mat ti = {
        .rows = t.rows,
        .cols = 2,
        .stride = t.stride,
        .es = &MAT_AT(t, 0, 0),
    };

    Mat to = {
        .rows = t.rows,
        .cols = 1,
        .stride = t.stride,
        .es = &MAT_AT(t, 0, ti.cols),
    };


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

    size_t epoch = 0;
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_R)) {
            epoch = 0;
            nn_rand(nn, -1, 1);
            plot.count = 0;
        }

        for (size_t i = 0; i < epochs_per_frame && !paused && epoch < max_epoch; ++i) {
            nn_backprop(nn, g, ti, to);
            nn_learn(nn, g, rate);
            epoch += 1;
            da_append(&plot, nn_cost(nn, ti, to));
        }

        BeginDrawing();
        Color background_color = {0x18, 0x18, 0x18, 0xFF};
        ClearBackground(background_color);
        {
            int w = GetRenderWidth();
            int h = GetRenderHeight();

            int rw = w/3;
            int rh = h*2/3;
            int rx = 0;
            int ry = h/2 - rh/2;

            gym_plot(plot, rx, ry, rw, rh);
            rx += rw;
            gym_render_nn(nn, rx, ry, rw, rh);
            rx += rw;
            verify_nn_gate(font, nn, rx, ry, rw, rh);

            char buffer[256];
            snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu, Rate: %f, Cost: %f", epoch, max_epoch, rate, nn_cost(nn, ti, to));
            DrawTextEx(font, buffer, CLITERAL(Vector2){}, h*0.04, 0, WHITE);
        }
        EndDrawing();
    }

    return 0;
}
