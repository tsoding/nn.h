#!/bin/sh

set -xe

clang -O3 -Wall -Wextra -o xor xor.c -lm
clang -O3 -Wall -Wextra -o adder adder.c -lm
