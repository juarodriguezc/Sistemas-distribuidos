//gcc pi_omp.c -o pi_omp -fopenmp
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

#define ITERATIONS 1e09
#define THREADS 16

int calculatePi(double *piTotal, int ID)
{   int start, end;   

    start = (ITERATIONS/omp_get_num_threads()) * ID;
    end = (ITERATIONS/omp_get_num_threads()) * (ID + 1);
    int i = start;

    do{
        *(piTotal + ID) = *(piTotal + ID) + (double)(4.0 / ((i*2)+1));
        i++;
        *(piTotal + ID) = *(piTotal + ID) - (double)(4.0 / ((i*2)+1));
        i++;
    }while(i < end);

    return 0;
}


int main()
{
    int i, threads = THREADS;
    double pi[threads];
    
    for(i = 0; i < THREADS; i++)
        pi[i] = 0;
        
    #pragma omp parallel num_threads(threads)
    {
        int ID = omp_get_thread_num();
        calculatePi(pi, ID);
    }
    
    for(i = 1; i < THREADS; i++)
    {
        pi[0] = pi[0] + pi[i];
    }
      
    printf("\npi: %2.10f \n", pi[0]);
    return 0;
}



