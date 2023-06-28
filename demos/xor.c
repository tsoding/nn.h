#define GYM_IMPLEMENTATION
#include "gym.h"

#define NN_IMPLEMENTATION
#include "nn.h"

size_t arch[] = {2, 2, 1};
size_t max_epoch = 100*1000;
size_t epochs_per_frame = 103;
float rate = 1.0f;
bool paused = true;

void verify_nn_gate(Font font, NN nn, Gym_Rect r)
{
    char buffer[256];
    float s = r.h*0.06;
    float pad = r.h*0.03;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = j;
            nn_forward(nn);
            snprintf(buffer, sizeof(buffer), "%zu @ %zu == %f", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
            DrawTextEx(font, buffer, CLITERAL(Vector2){r.x, r.y + (i*2 + j)*(s + pad)}, s, 0, WHITE);
        }
    }
}

int main(void)
{
    Region temp = region_alloc_alloc(256*1024*1024);

    Mat t = mat_alloc(NULL, 4, 3);
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

    NN nn = nn_alloc(NULL, arch, ARRAY_LEN(arch));
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
            NN g = nn_backprop(&temp, nn, ti, to);
            nn_learn(nn, g, rate);
            epoch += 1;
            da_append(&plot, nn_cost(nn, ti, to));
        }

        BeginDrawing();
        ClearBackground(GYM_BACKGROUND);
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
                gym_render_nn(nn, gym_layout_slot());
                verify_nn_gate(font, nn, gym_layout_slot());
            gym_layout_end();

            char buffer[256];
            snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu, Rate: %f, Cost: %f, Temporary Memory: %zu bytes", epoch, max_epoch, rate, nn_cost(nn, ti, to), region_occupied_bytes(&temp));
            DrawTextEx(font, buffer, CLITERAL(Vector2){}, h*0.04, 0, WHITE);
        }
        EndDrawing();

        region_reset(&temp);
    }

    return 0;
}
