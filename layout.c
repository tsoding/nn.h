#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <raylib.h>

typedef struct {
    float x;
    float y;
    float w;
    float h;
} Gym_Rect;

Gym_Rect gym_rect(float x, float y, float w, float h)
{
    Gym_Rect r = {0};
    r.x = x;
    r.y = y;
    r.w = w;
    r.h = h;
    return r;
}

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

typedef enum {
    GLO_HORZ,
    GLO_VERT,
} Gym_Layout_Orient;

typedef struct {
    Gym_Layout_Orient orient;
    Gym_Rect rect;
    size_t count;
    size_t i;
    float gap;
} Gym_Layout;

Gym_Rect gym_layout_slot_loc(Gym_Layout *l, const char *file_path, int line)
{
    if (l->i >= l->count) {
        fprintf(stderr, "%s:%d: ERROR: Layout overflow\n", file_path, line);
        exit(1);
    }

    Gym_Rect r = {0};

    switch (l->orient) {
    case GLO_HORZ:
        r.w = l->rect.w/l->count;
        r.h = l->rect.h;
        r.x = l->rect.x + l->i*r.w;
        r.y = l->rect.y;

        if (l->i == 0) { // First
            r.w -= l->gap/2;
        } else if (l->i >= l->count - 1) { // Last
            r.x += l->gap/2;
            r.w -= l->gap/2;
        } else { // Middle
            r.x += l->gap/2;
            r.w -= l->gap;
        }

        break;

    case GLO_VERT:
        r.w = l->rect.w;
        r.h = l->rect.h/l->count;
        r.x = l->rect.x;
        r.y = l->rect.y + l->i*r.h;

        if (l->i == 0) { // First
            r.h -= l->gap/2;
        } else if (l->i >= l->count - 1) { // Last
            r.y += l->gap/2;
            r.h -= l->gap/2;
        } else { // Middle
            r.y += l->gap/2;
            r.h -= l->gap;
        }

        break;

    default:
        assert(0 && "Unreachable");
    }

    l->i += 1;

    return r;
}

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

typedef struct {
    Gym_Layout *items;
    size_t count;
    size_t capacity;
} Gym_Layout_Stack;

void gym_layout_stack_push(Gym_Layout_Stack *ls, Gym_Layout_Orient orient, Gym_Rect rect, size_t count, float gap)
{
    Gym_Layout l = {0};
    l.orient = orient;
    l.rect = rect;
    l.count = count;
    l.gap = gap;
    da_append(ls, l);
}
#define gls_push gym_layout_stack_push

Gym_Rect gym_layout_stack_slot_loc(Gym_Layout_Stack *ls, const char *file_path, int line)
{
    assert(ls->count > 0);
    return gym_layout_slot_loc(&ls->items[ls->count - 1], file_path, line);
}

#define gym_layout_stack_slot(ls) gym_layout_stack_slot_loc(ls, __FILE__, __LINE__)
#define gls_slot gym_layout_stack_slot

void gym_layout_stack_pop(Gym_Layout_Stack *ls)
{
    assert(ls->count > 0);
    ls->count -= 1;
}
#define gls_pop gym_layout_stack_pop

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
