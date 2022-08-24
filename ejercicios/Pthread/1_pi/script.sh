#!/bin/bash
echo "-----------------------------------------------------------------------------"
echo "Computación paralela y distribuida - cálculo de PI"
echo "-----------------------------------------------------------------------------"
echo "Compilando programas ..."
sudo gcc -o calcpi_nopad calcpi_nopad.c -pthread
sudo gcc -o calcpi_pad calcpi_pad.c -pthread
echo "Compilación terminada, realizando pruebas ..."

echo "-----------------------------------------------------------------------------"
echo "                           Sin PAD                            "
echo "-----------------------------------------------------------------------------"
for ((c=1; c<=16; c*=2))
do
    ./calcpi_nopad $c
    ./calcpi_nopad $c
done

echo "-----------------------------------------------------------------------------"
echo "                           Con PAD                            "
echo "-----------------------------------------------------------------------------"
for ((c=1; c<=16; c*=2))
do
    ./calcpi_pad $c
    ./calcpi_pad $c
done

echo "-----------------------------------------------------------------------------"
printf "Pruebas terminadas, consulte el archivo 'times.csv' para un resumen de los tiempos\n"
echo "-----------------------------------------------------------------------------"
