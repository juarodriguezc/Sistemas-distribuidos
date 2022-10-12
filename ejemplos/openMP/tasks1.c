// Ejemplo que muestra el uso de tasks

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

int teclado()
{ char x;
  int i;
  for(i = 0; i < 10; i++){
    scanf(" %c", &x);
    printf("\nSe digitó: %c\n", x);
    if(x == 'q')  exit(0);
  }
}

int imprimir()
{
  int i;
  for(i = 0; i < 30; i++){
    printf(" %i", i);
    fflush(stdout);
    sleep(1);
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
      teclado();

    #pragma omp task
      imprimir();
  }
}
 return 0;
}
