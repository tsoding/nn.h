#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <raylib.h>

#define NN_IMPLEMENTATION
#define NN_ENABLE_GYM
#include "nn.h"

void widget(Gym_Rect r, Color c)
{
    Rectangle rr = {
        ceilf(r.x), ceilf(r.y), ceilf(r.w), ceilf(r.h)
    };
    if (CheckCollisionPointRec(GetMousePosition(), rr)) {
        c = ColorBrightness(c, 0.75f);
    }
    DrawRectangleRec(rr, c);
}

Color colors[] = {LIGHTGRAY, GRAY, DARKGRAY, YELLOW, GOLD, ORANGE, PINK, RED, MAROON, GREEN, LIME, DARKGREEN, SKYBLUE, BLUE, DARKBLUE, PURPLE, VIOLET, DARKPURPLE, BEIGE, BROWN, DARKBROWN, WHITE, MAGENTA, RAYWHITE};
#define colors_count (sizeof(colors)/sizeof(colors[0]))

size_t count = 0;

void go(Gym_Layout_Stack *ls, Gym_Rect r, Gym_Layout_Orient orient, size_t level)
{
    if (level >= 8) {
        widget(r, colors[rand()%colors_count]);
        count += 1;
        return;
    }

    size_t n = rand()%3 + 3;
    gls_push(ls, orient, r, n, 0);
    for (size_t i = 0; i < n; ++i) {
        go(ls, gls_slot(ls), 1 - orient, level + 1);
    }
    gls_pop(ls);
}

int main(void)
{
    size_t factor = 80;
    size_t width = 16*factor;
    size_t height = 9*factor;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(width, height, "Layout");
    SetTargetFPS(60);

    Gym_Layout_Stack ls = {0};

    bool once = false;

    while (!WindowShouldClose()) {
        float w = GetRenderWidth();
        float h = GetRenderHeight();
        float frame = h*0.05;
        float gap = 2.0f;

        BeginDrawing();
            ClearBackground(BLACK);
            Gym_Layout_Orient orient = GLO_HORZ;
            srand(69);
            go(&ls, gym_rect(0, frame, w, h - 2*frame), GLO_HORZ, 0);
            if (!once) {
                printf("Rect Count: %zu\n", count);
                once = true;
            }
        EndDrawing();

        assert(ls.count == 0);
    }

    CloseWindow();

    return 0;
}
