#!/bin/bash

echo "Build tt_metal libraries"
cd ./tt_metal
./build_all.sh
cd ..

echo "Build device libraries"
cd ./device
./build_all.sh
cd ..

echo "Build whisper libraries"
cd ./whisper
./build_all.sh
cd ..

echo "Build examples"
cd ./examples
./build_all.sh
cd ..


