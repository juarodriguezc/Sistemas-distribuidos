#!/bin/bash

echo "------------------------------------------------"
echo "          Sistemas distribuidos                 "
echo "------------------------------------------------"
echo ""

#Hacer make del deviceQuery modificado
cd deviceQuery
make >/dev/null
gpuInfo=$(./deviceQuery)
#Verificar que se tenga GPU Nvidia
if [ "$gpuInfo" = "-1" ]; then
    echo "La GPU del sistema no es compatible con CUDA"
else
    echo "La GPU del sistema es compatible con CUDA"

    mp=$(echo "$gpuInfo" | cut -d "_" -f 1)
    cores=$(echo "$gpuInfo" | cut -d "_" -f 2)
    name=$(echo "$gpuInfo" | cut -d "_" -f 3)


    echo "GPU: $name"
    echo "Se tienen $((mp)) multiprocesadores y $((cores)) cores por multiprocesador"
    echo ""
    echo "------------------------------------------------"


    cd ../


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
        ./MotionInterpolation src/video/test"$test"/ test.mp4 result.avi 80 0 $((2*$mp)) $cores
      done
    fi


fi


