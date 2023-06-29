#!/bin/sh

set -xe

CFLAGS="-O3 -Wall -Wextra -ggdb -I./thirdparty/ -I. `pkg-config --cflags raylib`"
LIBS="-lm `pkg-config --libs raylib` -lglfw -ldl -lpthread"

mkdir -p ./build/demos

clang $CFLAGS -o ./build/demos/xor demos/xor.c $LIBS
clang $CFLAGS -o ./build/demos/adder demos/adder.c $LIBS
clang $CFLAGS -o ./build/demos/img2nn demos/img2nn.c $LIBS
clang $CFLAGS -o ./build/demos/layout demos/layout.c $LIBS
clang $CFLAGS -o ./build/demos/shape demos/shape.c $LIBS
