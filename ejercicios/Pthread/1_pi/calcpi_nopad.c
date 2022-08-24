#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#include <sys/time.h>

#define ITERATIONS 2e09

double *piTotal;

int nThreads;

void *calculatePi(void *arg)
{
    int initIteration, endIteration, threadId = *(int *)arg;

    initIteration = (ITERATIONS / nThreads) * threadId;
    endIteration = initIteration + ((ITERATIONS / nThreads) - 1);
    // printf("\n %i  %i  %i", threadId, initIteration, endIteration);

    piTotal[threadId] = 0.0;
    do
    {
        piTotal[threadId] = piTotal[threadId] + (double)(4.0 / ((initIteration * 2) + 1));
        initIteration++;
        piTotal[threadId] = piTotal[threadId] - (double)(4.0 / ((initIteration * 2) + 1));
        initIteration++;
    } while (initIteration < endIteration);
    return 0;
}

int main(int argc, char *argv[])
{

    if ((argc - 1) != 1)
    {
        printf("Para una correcta ejecución: ./calcpi_nopad nThreads\n");
        exit(1);
    }
    nThreads = atoi(*(argv + 1));

    piTotal = malloc(nThreads * sizeof(double));

    int threadId[nThreads], i, *retval;
    pthread_t thread[nThreads];

    /*Declaración de variable para la escritura del archivo*/
    FILE *fp;

    /*Variables necesarias para medir tiempos*/
    struct timeval tval_before, tval_after, tval_result;

    /*Medición de tiempo de inicio*/
    gettimeofday(&tval_before, NULL);

    for (i = 0; i < nThreads; i++)
    {
        threadId[i] = i;
        pthread_create(&thread[i], NULL, (void *)calculatePi, &threadId[i]);
    }

    for (i = 0; i < nThreads; i++)
    {
        pthread_join(thread[i], (void **)&retval);
    }

    for (i = 1; i < nThreads; i++)
    {
        piTotal[0] = piTotal[0] + piTotal[i];
    }

    /*Medición de tiempo de finalización*/
    gettimeofday(&tval_after, NULL);

    /*Calcular los tiempos en tval_result*/
    timersub(&tval_after, &tval_before, &tval_result);

    /* Escribir los resultados en un csv*/
    fp = fopen("times.csv", "a");
    if (fp == NULL)
    {
        printf("Error al abrir el archivo \n");
        exit(1);
    }
    fprintf(fp, "%d,%ld.%06ld,False \n", nThreads, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    fclose(fp);

    printf("\npi: %2.10f, threads: %d, time: %ld.%06ld s, pad: False \n", piTotal[0], nThreads, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
}