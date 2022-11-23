#!/bin/bash

echo "------------------------------------------------"
echo "          Sistemas distribuidos                 "
echo "------------------------------------------------"
echo ""

echo "Verificando la ejecución de Cmake ..."
#Check if cmake command has already been executed
[ -e "cmake_install.cmake" ] && rm "cmake_install.cmake"
[ -e "CMakeCache.txt" ] && rm "CMakeCache.txt"
rm -rf "build"
rm -rf "Makefile"
rm -rf "CMakeFiles"
cmake .
echo "Compilando el programa ..."
make

if [ $? -eq 0 ]; then
  echo "Compilación terminada"
fi