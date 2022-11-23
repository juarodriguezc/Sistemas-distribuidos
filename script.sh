#!/bin/bash
echo "Ejecutando el programa ..."
for test in {1,4}
do
  for (( i=1; i<=(16); i=i*2 ))
  do
    echo ""
    #mpirun -np $i --hostfile mpi_hosts ./MotionInterpolation /src/video/test"$test"/ test.mp4 result.avi 80 0
  done
done
mpirun -np 4 --hostfile mpi_hosts ./MotionInterpolation /home/"$USER"/Sistemas-distribuidos/src/video/test4/ test.mp4 result.avi 8 0