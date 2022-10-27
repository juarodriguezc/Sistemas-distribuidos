#!/bin/bash

echo "------------------------------------------------"
echo "          Sistemas distribuidos                 "
echo "------------------------------------------------"
echo ""

echo "Verificando la ejecución de Cmake ..."
#Check if cmake command has already been executed
if [ ! -f "MotionInterpolation" ]; then
  echo "$DIRECTORY does not exist."
  [ -e "cmake_install.cmake" ] && rm "cmake_install.cmake"
  [ -e "CMakeCache.txt" ] && rm "CMakeCache.txt"
  rm -rf "build"
  rm -rf "Makefile"
  rm -rf "CMakeFiles"
  cmake .
  else
  echo "$DIRECTORY does exist."
fi
echo "Compilando el programa ..."
make
echo "Ejecutando el programa"

./MotionInterpolation src/video/test2/test.mp4 src/video/test2/result.avi 8