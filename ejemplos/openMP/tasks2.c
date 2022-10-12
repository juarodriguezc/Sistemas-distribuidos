// Ejemplo que muestra el uso de tasks
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#define ITER 1e09

int xfunc()
{ double x;
  int i;
  for(i = 0; i < ITER; i++){
    x = x + sin(i);
  }
}

int yfunc(int a)
{
  double x;
  int i;
  for(i = 0; i < ITER; i++){
    x = x + cos(i);
  }
}

int main()
{

printf("\nejemplo de uso de tasks");
fflush(stdout);

#pragma omp parallel
{
#pragma omp single
  {
    #pragma omp task
      xfunc();

    #pragma omp task
      yfunc();

    #pragma omp task
      xfunc();

    #pragma omp task
      yfunc();

  }
}
 return 0;
}
