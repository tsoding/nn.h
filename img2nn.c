#include <assert.h>
#include <stdio.h>
#include <float.h>

#include <raylib.h>
#include <raymath.h>

#include "stb_image.h"
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#define NN_ENABLE_GYM
#include "nn.h"

size_t arch[] = {3, 11, 11, 9, 1};
size_t epoch = 0;
size_t max_epoch = 100*1000;
size_t batches_per_frame = 200;
size_t batch_size = 28;
float rate = 1.0f;
bool paused = true;

char *args_shift(int *argc, char ***argv)
{
    assert(*argc > 0);
    char *result = **argv;
    (*argc) -= 1;
    (*argv) += 1;
    return result;
}

int main(int argc, char **argv)
{
    const char *program = args_shift(&argc, &argv);

    if (argc <= 0) {
        fprintf(stderr, "Usage: %s <image1> <image2>\n", program);
        fprintf(stderr, "ERROR: no image1 is provided\n");
        return 1;
    }
    const char *img1_file_path = args_shift(&argc, &argv);

    if (argc <= 0) {
        fprintf(stderr, "Usage: %s <image1> <image2>\n", program);
        fprintf(stderr, "ERROR: no image2 is provided\n");
        return 1;
    }
    const char *img2_file_path = args_shift(&argc, &argv);

    int img1_width, img1_height, img1_comp;
    uint8_t *img1_pixels = (uint8_t *)stbi_load(img1_file_path, &img1_width, &img1_height, &img1_comp, 0);
    if (img1_pixels == NULL) {
        fprintf(stderr, "ERROR: could not read image %s\n", img1_file_path);
        return 1;
    }
    if (img1_comp != 1) {
        fprintf(stderr, "ERROR: %s is %d bits image. Only 8 bit grayscale images are supported\n", img1_file_path, img1_comp*8);
        return 1;
    }

    int img2_width, img2_height, img2_comp;
    uint8_t *img2_pixels = (uint8_t *)stbi_load(img2_file_path, &img2_width, &img2_height, &img2_comp, 0);
    if (img2_pixels == NULL) {
        fprintf(stderr, "ERROR: could not read image %s\n", img2_file_path);
        return 1;
    }
    if (img2_comp != 1) {
        fprintf(stderr, "ERROR: %s is %d bits image. Only 8 bit grayscale images are supported\n", img2_file_path, img2_comp*8);
        return 1;
    }

    printf("%s size %dx%d %d bits\n", img1_file_path, img1_width, img1_height, img1_comp*8);
    printf("%s size %dx%d %d bits\n", img2_file_path, img2_width, img2_height, img2_comp*8);

    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));

    Mat t = mat_alloc(img1_width*img1_height + img2_width*img2_height, NN_INPUT(nn).cols + NN_OUTPUT(nn).cols);
    for (int y = 0; y < img1_height; ++y) {
        for (int x = 0; x < img1_width; ++x) {
            size_t i = y*img1_width + x;
            MAT_AT(t, i, 0) = (float)x/(img1_width - 1);
            MAT_AT(t, i, 1) = (float)y/(img1_height - 1);
            MAT_AT(t, i, 2) = 0.f;
            MAT_AT(t, i, 3) = img1_pixels[i]/255.f;
        }
    }

    for (int y = 0; y < img2_height; ++y) {
        for (int x = 0; x < img2_width; ++x) {
            size_t i = img1_width*img1_height + y*img2_width + x;
            MAT_AT(t, i, 0) = (float)x/(img2_width - 1);
            MAT_AT(t, i, 1) = (float)y/(img2_height - 1);
            MAT_AT(t, i, 2) = 1.f;
            MAT_AT(t, i, 3) = img2_pixels[y*img2_width + x]/255.f;
        }
    }

    nn_rand(nn, -1, 1);

    size_t WINDOW_FACTOR = 80;
    size_t WINDOW_WIDTH = (16*WINDOW_FACTOR);
    size_t WINDOW_HEIGHT = (9*WINDOW_FACTOR);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "gym");
    SetTargetFPS(60);

    Plot plot = {0};

    size_t preview_width = 28;
    size_t preview_height = 28;

    Image preview_image1 = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D preview_texture1 = LoadTextureFromImage(preview_image1);

    Image preview_image2 = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D preview_texture2 = LoadTextureFromImage(preview_image2);

    Image preview_image3 = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D preview_texture3 = LoadTextureFromImage(preview_image3);

    Image original_image1 = GenImageColor(img1_width, img1_height, BLACK);
    for (size_t y = 0; y < (size_t) img1_height; ++y) {
        for (size_t x = 0; x < (size_t) img1_width; ++x) {
            uint8_t pixel = img1_pixels[y*img1_width + x];
            ImageDrawPixel(&original_image1, x, y, CLITERAL(Color) { pixel, pixel, pixel, 255 });
        }
    }
    Texture2D original_texture1 = LoadTextureFromImage(original_image1);

    Image original_image2 = GenImageColor(img2_width, img2_height, BLACK);
    for (size_t y = 0; y < (size_t) img2_height; ++y) {
        for (size_t x = 0; x < (size_t) img2_width; ++x) {
            uint8_t pixel = img2_pixels[y*img2_width + x];
            ImageDrawPixel(&original_image2, x, y, CLITERAL(Color) { pixel, pixel, pixel, 255 });
        }
    }
    Texture2D original_texture2 = LoadTextureFromImage(original_image2);

    Gym_Batch gb = {0};

    float scroll = 0.5f;
    bool scroll_dragging = false;

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
            gym_process_batch(&gb, batch_size, nn, g, t, rate);
            if (gb.finished) {
                epoch += 1;
                da_append(&plot, gb.cost);
                mat_shuffle_rows(t);
            }
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

            float scale = rh*0.01;

            for (size_t y = 0; y < (size_t) preview_height; ++y) {
                for (size_t x = 0; x < (size_t) preview_width; ++x) {
                    MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(preview_width - 1);
                    MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(preview_height - 1);
                    MAT_AT(NN_INPUT(nn), 0, 2) = 0.f;
                    nn_forward(nn);
                    uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
                    ImageDrawPixel(&preview_image1, x, y, CLITERAL(Color) { pixel, pixel, pixel, 255 });
                }
            }

            for (size_t y = 0; y < (size_t) preview_height; ++y) {
                for (size_t x = 0; x < (size_t) preview_width; ++x) {
                    MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(preview_width - 1);
                    MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(preview_height - 1);
                    MAT_AT(NN_INPUT(nn), 0, 2) = 1.f;
                    nn_forward(nn);
                    uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
                    ImageDrawPixel(&preview_image2, x, y, CLITERAL(Color) { pixel, pixel, pixel, 255 });
                }
            }

            for (size_t y = 0; y < (size_t) preview_height; ++y) {
                for (size_t x = 0; x < (size_t) preview_width; ++x) {
                    MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(preview_width - 1);
                    MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(preview_height - 1);
                    MAT_AT(NN_INPUT(nn), 0, 2) = scroll;
                    nn_forward(nn);
                    uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
                    ImageDrawPixel(&preview_image3, x, y, CLITERAL(Color) { pixel, pixel, pixel, 255 });
                }
            }

            UpdateTexture(preview_texture1, preview_image1.data);
            DrawTextureEx(preview_texture1, CLITERAL(Vector2) { rx, ry  + img1_height*scale }, 0, scale, WHITE);
            DrawTextureEx(original_texture1, CLITERAL(Vector2) { rx, ry}, 0, scale, WHITE);

            UpdateTexture(preview_texture2, preview_image2.data);
            DrawTextureEx(preview_texture2, CLITERAL(Vector2) { rx + img1_width*scale, ry  + img2_height*scale }, 0, scale, WHITE);
            DrawTextureEx(original_texture2, CLITERAL(Vector2) { rx + img1_width*scale, ry}, 0, scale, WHITE);

            UpdateTexture(preview_texture3, preview_image3.data);
            DrawTextureEx(preview_texture3, CLITERAL(Vector2) { rx, ry + img2_height*scale*2 }, 0, 2*scale, WHITE);

            {
                float pad = rh*0.05;
                Vector2 size = { img1_width*scale*2, rh*0.02 };
                Vector2 position = { rx, ry + img2_height*scale*4 + pad };
                DrawRectangleV(position, size, WHITE);
                float knob_radius = rh*0.03;
                Vector2 knob_position = { rx + size.x*scroll, position.y + size.y*0.5f};
                DrawCircleV(knob_position, knob_radius, RED);

                if (scroll_dragging) {
                    float x = GetMousePosition().x;
                    if (x < position.x) x = position.x;
                    if (x > position.x + size.x) x = position.x + size.x;
                    scroll = (x - position.x)/size.x;
                }

                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                    Vector2 mouse_position = GetMousePosition();
                    if (Vector2Distance(mouse_position, knob_position) <= knob_radius) {
                        scroll_dragging = true;
                    }
                }

                if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
                    scroll_dragging = false;
                }
            }

            char buffer[256];
            snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu, Rate: %f, Cost: %f", epoch, max_epoch, rate, plot.count > 0 ? plot.items[plot.count - 1] : 0);
            DrawText(buffer, 0, 0, h*0.04, WHITE);
        }
        EndDrawing();
    }

    for (size_t y = 0; y < (size_t) img1_height; ++y) {
        for (size_t x = 0; x < (size_t) img1_width; ++x) {
            uint8_t pixel = img1_pixels[y*img1_width + x];
            if (pixel) printf("%3u ", pixel); else printf("    ");
        }
        printf("\n");
    }

    for (size_t y = 0; y < (size_t) img1_height; ++y) {
        for (size_t x = 0; x < (size_t) img1_width; ++x) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(img1_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(img1_height - 1);
            MAT_AT(NN_INPUT(nn), 0, 2) = 0.5f;
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
            if (pixel) printf("%3u ", pixel); else printf("    ");
        }
        printf("\n");
    }

    size_t out_width = 512;
    size_t out_height = 512;
    uint8_t *out_pixels = malloc(sizeof(*out_pixels)*out_width*out_height);
    assert(out_pixels != NULL);

    for (size_t y = 0; y < out_height; ++y) {
        for (size_t x = 0; x < out_width; ++x) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(out_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(out_height - 1);
            MAT_AT(NN_INPUT(nn), 0, 2) = 0.5f;
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
            out_pixels[y*out_width + x] = pixel;
        }
    }

    const char *out_file_path = "upscaled.png";
    if (!stbi_write_png(out_file_path, out_width, out_height, 1, out_pixels, out_width*sizeof(*out_pixels))) {
        fprintf(stderr, "ERROR: could not save image %s\n", out_file_path);
        return 1;
    }

    printf("Generated %s from %s\n", out_file_path, img1_file_path);

    return 0;
}
