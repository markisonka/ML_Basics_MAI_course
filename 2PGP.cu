#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>

using namespace std;

#define CSC(call)                                          \
do {                                                \
    cudaError_t res = call;                                \
    if (res != cudaSuccess) {                                \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));        \
        exit(0);                                    \
    }                                            \
} while(0)

// __device__  - выполняется на gpu, вызывается с gpu
__device__ int find_perfect(int* arr,res* res,) {
    if (num > 1) {
        //printf("find perfect initiated \n");
        int sum = 0;
        for (int i = 1; i < (num); i++) {
            if (num % i == 0)
                sum += i;
        }
        //printf("%d ", num);
        //printf("%d \n", sum);
        if (sum == num) {
            //printf("Calculation worked \n");
            return sum;
        }
        else return 0;
    }
    else return 0;
}



// __global__  - выполняется на gpu, вызывается с cpu
__global__ void kernel(int* arr, int* ans, int n) { //отличное от C++ (__global__)
    int i, idx = blockDim.x * blockIdx.x + threadIdx.x;            // Абсолютный номер потока 
    //blockDim (кол-во потоков в блоке) и blockIdx (номер блока в thread) - глобальные константы в CUDA, по осям x,y,z и т.д
    //threadIds (номер thread-a) 
    int offset = blockDim.x * gridDim.x;                        // Общее кол-во потоков

    /*
    присвоить переменной, в которой будет накапливаться сумма делителей, 0.
    В цикле от 1 до половины текущего натурального числа
    пытаться разделить исследуемое число нацело на счетчик внутреннего цикла.
    Если делитель делит число нацело, то добавить его к переменной суммы делителей.
    Если сумма делителей равна исследуемому натуральному числу, то это число совершенно и следует вывести его на экран.
    */

    //printf("Method launched \n");
    for (i = idx; i < n; i += offset) // Для всех требование - внутри цикла for()
    // Данных в тестах больше, чем поток, который можно выделить => нужно завернуть в цикл
    // offset - суммарное кол-во выделенных потоков на обработку
    // без него будет падать
        ans = find_perfect(arr);
}

__global__ void kernel(int* arr, int* ans, int n) { //отличное от C++ (__global__)
    int i, idx = blockDim.x * blockIdx.x + threadIdx.x;            // Абсолютный номер потока 
    //blockDim (кол-во потоков в блоке) и blockIdx (номер блока в thread) - глобальные константы в CUDA, по осям x,y,z и т.д
    //threadIds (номер thread-a) 
    int offset = blockDim.x * gridDim.x;                        // Общее кол-во потоков

    /*
    присвоить переменной, в которой будет накапливаться сумма делителей, 0.
    В цикле от 1 до половины текущего натурального числа
    пытаться разделить исследуемое число нацело на счетчик внутреннего цикла.
    Если делитель делит число нацело, то добавить его к переменной суммы делителей.
    Если сумма делителей равна исследуемому натуральному числу, то это число совершенно и следует вывести его на экран.
    */

    //printf("Method launched \n");
    for (i = idx; i < n; i += offset) // Для всех требование - внутри цикла for()
    // Данных в тестах больше, чем поток, который можно выделить => нужно завернуть в цикл
    // offset - суммарное кол-во выделенных потоков на обработку
    // без него будет падать
	
        ans = find_perfect(arr);
}


__global__ void matMult ( float * a, float * b, int n, float * c ) {
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	int aBegin = n * BLOCK_SIZE * by;
	int aEnd = aBegin + n - 1;
	int bBegin = BLOCK_SIZE * bx;
	int aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;
	float sum = 0.0f;
	for ( int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep ){
	__shared__ float as [BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float bs [BLOCK_SIZE][BLOCK_SIZE];
	as [ty][tx] = a [ia + n * ty + tx];
	bs [ty][tx] = b [ib + n * ty + tx];
	__syncthreads (); // Synchronize to make sure the matrices are loaded
	for ( int k = 0; k < BLOCK_SIZE; k++ )
		sum += as [ty][k] * bs [k][tx];
		__syncthreads (); // Synchronize to make sure submatrices not needed
	}
	c [n * BLOCK_SIZE * by + BLOCK_SIZE * bx + n * ty + tx] = sum;
}


int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;

    size_t size = n * sizeof(int);

    int* N_array = (int*)malloc(size); //выделение массива 1
    int* answer = (int*)malloc(size); //выделение массива 2
    for (int i = 0; i < n; i++)
        N_array[i] = i; //заполнение массива 1

    for (int i = 0; i < n; i++)
        answer[i] = 0; //заполнение массива 1
    

    int* dev_arr1;
    CSC(cudaMalloc(&dev_arr1, size)); //выделение массива на устройстве 
    CSC(cudaMemcpy(dev_arr1, N_array, size, cudaMemcpyHostToDevice));

    int* result;
    CSC(cudaMalloc(&result, size)); //выделение массива на устройстве 
    CSC(cudaMemcpy(result, answer, size, cudaMemcpyHostToDevice));

    kernel << < 2*n-1, 1 >> > (dev_arr1, result, n); //отличное от C++ (<<<>>>), стандартная функция
    // Многопоточное
    // 256 блоков и 256 потоков(Thread)



    CSC(cudaMemcpy(answer, result, size, cudaMemcpyDeviceToHost));
    CSC(cudaFree(result));

    for (int i = 0; i < n; i++)
        if (answer[i] > 0) {
            cout << answer[i] << ' ';
        }
    cout << endl;
    free(N_array);
    free(answer);
    return 0;
}


for(i=0,i<n,i++)
{res[0,i]=add[0,i]
res[i,0]=

}
__syncthreads ();
for (i=2, i<=(n-1)*2, i++) 
	{
	 if(i<=n):
		{
		for(j=0, j<i-1)
			{res[1+j;i-j]=RaschLU(add,res,1+j;i-j,n,i)}
		}
	__syncthreads ();
	}





//переменная n(размерность иходной ''квадратной'' матрицы) должна получить значение до этого момента
            double[,] A = new double[n, n];
            double[,] L = new double[n, n];
            double[,] U = new double[n, n];
//до этого момента массив A должен быть полностью определен
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    U[0, i] = A[0, i]
                    L[i, 0] = A[i, 0] / U[0, 0];
                    double sum = 0;
                    for (int k = 0; k < i; k++)
                    {
                        sum += L[i, k] * U[k, j];
                    }
                    U[i, j] = A[i, j] - sum;
                    if (i > j)
                    {
                        L[j, i] = 0;
                    }
                    else
                    {
                        sum = 0;
                        for (int k = 0; k < i; k++)
                        {
                            sum += L[j, k] * U[k, i];
                        }
                        L[j, i] = (A[j, i] - sum) / U[i, i];
                    }
                }
            }
//после выполнения цикла в массиве L - элементы матрицы L, в массиве U - элементы матрицы U.

//Теперь можно вычислять определитель