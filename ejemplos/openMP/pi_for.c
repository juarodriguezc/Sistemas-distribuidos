#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define MAX_TERMS 2000000000

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("usage: ./pi thread_count");
        return 0;
    }
    
    int thread_count = atoi(argv[1]);
    
    double result = 0;
    
    #pragma omp parallel for num_threads(thread_count) reduction(+:result)
    for(int i = 0; i < MAX_TERMS; ++i) {
      result += 4.0 * (i % 2 == 0 ? 1: -1) / (2.0 * i + 1);
    }
    
    printf("Result: %.16lf\n", result);
    return 0;
}
