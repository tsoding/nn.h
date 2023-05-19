#!/bin/sh

set -xe

CFLAGS="-O3 -Wall -Wextra"
LIBS="-lm"

clang $CFLAGS -o adder_gen adder_gen.c $LIBS
clang $CFLAGS -o xor_gen xor_gen.c $LIBS
clang $CFLAGS `pkg-config --cflags raylib` -o gym gym.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread
clang $CFLAGS `pkg-config --cflags raylib` -o img2mat img2mat.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread
clang $CFLAGS -o bench bench.c
