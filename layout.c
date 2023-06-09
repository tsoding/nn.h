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

int main(void)
{
    size_t factor = 80;
    size_t width = 16*factor;
    size_t height = 9*factor;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(width, height, "Layout");
    SetTargetFPS(60);

    Gym_Layout_Stack ls = {0};

    while (!WindowShouldClose()) {
        float w = GetRenderWidth();
        float h = GetRenderHeight();
        float frame = h*0.15;
        float gap = 10.0f;

        BeginDrawing();
            ClearBackground(BLACK);
            gls_push(&ls, GLO_HORZ, gym_rect(0, frame, w, h - 2*frame), 3, gap);
                widget(gls_slot(&ls), RED);
                widget(gls_slot(&ls), BLUE);
                gls_push(&ls, GLO_VERT, gls_slot(&ls), 3, gap);
                    gls_push(&ls, GLO_HORZ, gls_slot(&ls), 2, gap);
                        gls_push(&ls, GLO_VERT, gls_slot(&ls), 2, gap);
                           widget(gls_slot(&ls), GREEN);
                           gls_push(&ls, GLO_HORZ, gls_slot(&ls), 2, gap);
                              widget(gls_slot(&ls), GREEN);
                              widget(gls_slot(&ls), GREEN);
                           gls_pop(&ls);
                        gls_pop(&ls);
                        widget(gls_slot(&ls), PURPLE);
                    gls_pop(&ls);
                    gls_push(&ls, GLO_HORZ, gls_slot(&ls), 3, gap);
                        widget(gls_slot(&ls), YELLOW);
                        widget(gls_slot(&ls), YELLOW);
                        widget(gls_slot(&ls), YELLOW);
                    gls_pop(&ls);
                    widget(gls_slot(&ls), PURPLE);
                gls_pop(&ls);
            gls_pop(&ls);
        EndDrawing();

        assert(ls.count == 0);
    }

    CloseWindow();

    return 0;
}
