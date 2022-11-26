#!/bin/bash
echo "Ejecutando el programa ..."
for test in {1,4}
do
  for (( process=1; process<=(8); process=process*2 ))
  do
    for threads in {1,2,3,4}
    do
      echo " Test #$test  -  #Process: $process:   -   #Threads: $threads"
      mpirun -np $process --hostfile mpi_hosts ./MotionInterpolation /home/"$USER"/Sistemas-distribuidos/src/video/test"$test"/ test.mp4 result.avi 80 0 $threads
      #mpirun -np $process --hostfile mpi_hosts ./MotionInterpolation src/video/test"$test"/ test.mp4 result.avi 10 0 $threads
    done   
  done
done
#mpirun -np 4 --hostfile mpi_hosts ./MotionInterpolation src/video/test5/ test.mp4 result.avi 80 0 2
