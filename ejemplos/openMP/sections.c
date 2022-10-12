#include <stdio.h>
#include <omp.h>
#include <math.h>

void XAXIS()
{
    int i;
    double x;
    printf("x axis");
    for(i = 0; i < 3e07; i++)
        x = x + sin(x);
}

void YAXIS()
{
    int i;
    double x;
    printf("y axis");
    for(i = 0; i < 2e07; i++)
        x = x + sin(x);
}

void ZAXIS()
{
    int i;
    double x;
    printf("z axis");
    for(i = 0; i < 1e07; i++)
        x = x + sin(x);
}


int main()
{

#pragma omp parallel sections
{
    #pragma omp section
        XAXIS();

    #pragma omp section
        YAXIS();

    #pragma omp section
        ZAXIS();
}
}
