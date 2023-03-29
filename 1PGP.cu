#include "cuda_runtime.h"      // CUDA runtime
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime> 
//#include <vector> 

#define  N 16
#define BLOCK_SIZE 16


__global__ void mult( float *a, float *b, float *c) { 
    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    float sum = 0.0f; 
    int ia = N * BLOCK_SIZE * by + N * ty; // смещение для a[i][0]
    int ib = BLOCK_SIZE * bx + tx;  //  смещение для b[0][j]
    
	for (int k=0; k < N; k++)   // вычисляем элемент
           sum += a[ia + k] * b[ib + k * N]; 
   	int ic = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;  //   смещение для элемента c
	c[ic + N * ty + tx] = sum;    //  запоминаем результат
} 



int main()
{
	printf("\n  N = %i\n\n", N );
	
	float *a,*b,*c; 
	int numBytes = N*N*sizeof(float);
	a=(float *)malloc(numBytes); 
	b=(float *)malloc(numBytes); 
	c=(float *)malloc(numBytes); 
	
	//// Тестовый массив 4x4
	a[0] = 2.0f; a[1] = 1.0f; a[2] = 3.0f; a[3] = 5.0f; 
	a[4] = 0.0f; a[5] = 5.0f; a[6] = 0.0f; a[7] = 2.0f; 
	a[8] = 1.0f; a[9] = 2.0f; a[10] = 3.0f; a[11] = 1.0f; 
	a[12] = 3.0f;	a[13] = 4.0f; a[14] = 1.0f; a[15] = 4.0f;
	
	b[0] = 4.0f; b[1] = 1.0f; b[2] = 0.0f; b[3] = 1.0f; 
	b[4] = 0.0f; b[5] = 3.0f; b[6] = 2.0f; b[7] = 4.0f; 
	b[8] = 2.0f; b[9] = 4.0f; b[10] = 5.0f; b[11] = 0.0f; 
	b[12] = 2.0f; b[13] = 5.0f; b[14] = 4.0f; b[15] = 1.0f;
	 

	// Единичные a и b 
	//for (int i=0; i<N*N; i++) { 
       //   a[i]=1.0f; 
       //   b[i]=1.0f; 
	//}

	for (int i=0; i<N*N; i++) c[i]=0.0f; 
    
	// создать элементы strat и stop для событий
       cudaEvent_t start, stop;
       float gpuTime = 0.0f;
	cudaEventCreate ( &start );
	cudaEventCreate ( &stop );
	cudaEventRecord ( start, 0 );    
	float *dev_a, *dev_b, *dev_c; 
    
	//  Выделить память в DRAM
	cudaMalloc((void**) &dev_a, numBytes); 
	cudaMalloc((void**) &dev_b, numBytes); 
	cudaMalloc((void**) &dev_c, numBytes); 

	// Скопировать из CPU в DRAM
	cudaMemcpy(dev_a, a,numBytes,cudaMemcpyHostToDevice); 
	cudaMemcpy(dev_b, b,numBytes,cudaMemcpyHostToDevice); 
	cudaMemcpy(dev_c, c,numBytes,cudaMemcpyHostToDevice); 
    
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(N/threads.x, N/threads.y);
	

	//  Вызвать ядро
	mult<<<blocks, threads>>>(dev_a,dev_b,dev_c); 
	
	cudaThreadSynchronize();  // Дождаться окончания расчета
	cudaMemcpy(c, dev_c,numBytes,cudaMemcpyDeviceToHost); 

	// get data back
	cudaEventRecord ( stop, 0 );

    // force synchronization
	cudaEventSynchronize ( stop );
	cudaEventElapsedTime ( &gpuTime, start, stop );

  
printf("\n===========   GPU     =====================\n");

	printf("\n"); 
	for (int i=0; i<N*N; i++) { 
        if (i%N==0) printf("\n"); 
        printf("%.0f ",a[i]); 
    } 
    printf("\n"); 
    for (int i=0; i<N*N; i++) { 
        if (i%N==0) printf("\n"); 
        printf("%.0f ",b[i]); 
    } 
    printf("\n"); 
    for (int i=0; i<N*N; i++) { 
        if (i%N==0) printf("\n"); 
        printf("%.0f ",c[i]); 
    } 
	printf("\n\n"); 


	// print the GPU times
	printf("DEVICE GPU compute time: %.2f milliseconds\n\n", gpuTime );


    cudaFree( dev_a); 
    cudaFree( dev_b); 
    cudaFree( dev_c); 

    free(a); 
    free(b); 
    free(c); 

	system("pause");
	cudaProfilerStop();
    return 0;
}
