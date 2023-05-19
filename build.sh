#!/bin/sh

set -xe

CFLAGS="-O3 -Wall -Wextra"
LIBS="-lm"
CC=clang

$CC $CFLAGS -o adder_gen adder_gen.c $LIBS
$CC $CFLAGS -o xor_gen xor_gen.c $LIBS
$CC $CFLAGS `pkg-config --cflags raylib` -o gym gym.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread
$CC $CFLAGS `pkg-config --cflags raylib` -o img2mat img2mat.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread
$CC $CFLAGS -o bench bench.c $LIBS
