#!/bin/sh

set -xe

CFLAGS="-O3 -Wall -Wextra `pkg-config --cflags raylib`"
LIBS="`pkg-config --libs raylib` -lm -lglfw -ldl -lpthread"

clang $CFLAGS -o adder_gen adder_gen.c $LIBS
clang $CFLAGS -o xor_gen xor_gen.c $LIBS
clang $CFLAGS -o gym gym.c $LIBS
clang $CFLAGS -o bench bench.c
