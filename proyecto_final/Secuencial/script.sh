#!/bin/bash

echo "------------------------------------------------"
echo "          Sistemas distribuidos                 "
echo "------------------------------------------------"
echo ""

echo "Verificando la ejecución de Cmake ..."
#Check if cmake command has already been executed
if [ ! -f "MotionInterpolation" ]; then
  [ -e "cmake_install.cmake" ] && rm "cmake_install.cmake"
  [ -e "CMakeCache.txt" ] && rm "CMakeCache.txt"
  rm -rf "build"
  rm -rf "Makefile"
  rm -rf "CMakeFiles"
  cmake .
fi
echo "Compilando el programa ..."
make

if [ $? -eq 0 ]; then
  echo "Compilación terminada"
  echo "Ejecutando el programa ..."
  for test in {1,4}
  do
    echo "TEST #$test \n"
    ./MotionInterpolation src/video/test"$test"/ test.mp4 result.avi 25 0
  done
fi
#./MotionInterpolation src/video/test4/ test.mp4 result.avi 80 1 16
