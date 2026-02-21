#!/bin/bash

# Ensure CMake is installed
sudo apt-get update
sudo apt-get install -y cmake

# Create Build Directory
mkdir build

# Configure CMake
cmake -S . -B build

# Build Project
cmake --build build

# Run Tests
cd build && ctest --output-on-failure
