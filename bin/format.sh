#!/bin/bash

# Get current directory
BASEDIR=$(dirname "$0")

# C++ formatting
clang-format -i $BASEDIR/../src/*.cpp
clang-format -i $BASEDIR/../src/*.h
# clang-format -i $BASEDIR/../tests/*.cpp
