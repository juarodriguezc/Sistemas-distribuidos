#!/bin/bash

echo "------------------------------------------------"
echo "------------------------------------------------"
echo "             Sistemas distribuidos              "
echo "------------------------------------------------"
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

    mp=$((10#$(echo "$gpuInfo" | cut -d "_" -f 1)))
    cores=$((10#$(echo "$gpuInfo" | cut -d "_" -f 2)))
    name=$(echo "$gpuInfo" | cut -d "_" -f 3)

    cd ../

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
      echo "✓ Compilación terminada"

      echo "************************************************"
      echo "GPU: $name"
      echo "Se tienen $((mp)) multiprocesadores y $((cores)) cores por multiprocesador"
      echo "************************************************"
      echo ""

      echo "Ejecutando el programa ..."

      for test in {1,4}
      do
        for (( i=1; i<=((2*$mp)); i=i*2 ))
        do
          for (( j=4; j<=((2*$cores)); j=j*2 ))
          do
            ./MotionInterpolation src/video/test"$test"/ test.mp4 result.avi 25 0 $i $j
            if [ $? -eq 0 ]; then
              echo "✓ Interpolación de video realizada correctamente"
            else
                echo "Error en la ejecución del programa :("
                exit 1
            fi
          done
        done
      done
    else
      echo "Error en la compilación del programa :("
      exit 1
    fi
fi
echo "------------------------------------------------"


