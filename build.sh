#!/bin/sh

set -xe

CFLAGS="-O3 -Wall -Wextra -I./thirdparty/"
LIBS="-lm"

clang $CFLAGS -o adder_gen adder_gen.c $LIBS
clang $CFLAGS `pkg-config --cflags raylib` -o xor xor.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread
clang $CFLAGS `pkg-config --cflags raylib` -o gym gym.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread
clang $CFLAGS `pkg-config --cflags raylib` -o img2nn img2nn.c $LIBS `pkg-config --libs raylib` -lglfw -ldl -lpthread
