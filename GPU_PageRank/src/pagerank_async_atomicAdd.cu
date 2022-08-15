#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>

const float alpha = 0.85;
// __device__ __managed__ double epsilon = 0.00015;
// struct Lock {
//     int *mutex;
//     Lock(void) {
//         int state = 0;
//         cudaMalloc((void **) &mutex, sizeof(int));
//         cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
//     }
//     ~Lock(void) {
//         cudaFree(mutex);
//     }
//     __device__ void lock(void) {
//         while (atomicCAS(mutex, 0, 1) != 0);
//     }
//     __device__ void unlock(void) {
//         atomicExch(mutex, 0);
//     }
// };

__global__ void computeValue(const int nodes, float* value, float* new_value, const int* rowdeg, const int* colptr, const int* row, const int* col, int* is_valid_nodes)
{
    extern __shared__ float tile[];
    tile[threadIdx.x] = 1 - alpha;
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < nodes && is_valid_nodes[tid] == 1) {
        int e = e = colptr[tid];
        while (e < colptr[tid + 1]) {
            tile[threadIdx.x] += alpha * value[row[e]] / (float)rowdeg[row[e]];
            e++;
        }

        new_value[tid] = tile[threadIdx.x];
    }
}

__global__ void updateActive(const int nodes, float* value, float* new_value, int* is_valid_nodes, unsigned int* num_active_nodes_d, const double epsilon)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < nodes && is_valid_nodes[tid] == 1) {
        if(abs(value[tid] - new_value[tid]) <= epsilon)
        {
            is_valid_nodes[tid] = 0;
            int old = atomicAdd(num_active_nodes_d, -1);
        }
    }
}

void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* colptr, const int* row, const int* col, const double epsilon, const int threads_per_block)
{
    float *d_value, *d_new_value;
    int *is_valid_nodes, *d_is_valid_nodes, *d_rowdeg, *d_colptr, *d_row, *d_col;
    unsigned int num_active_nodes = nodes;
    unsigned int * num_active_nodes_d;


    cudaMalloc(&d_value, sizeof(float) * nodes);
    cudaMalloc(&d_new_value, sizeof(float) * nodes);

    cudaMalloc(&d_is_valid_nodes, sizeof(int) * nodes);
    is_valid_nodes = (int*)malloc(sizeof(int) * nodes);
    assert(is_valid_nodes != NULL);
    for (int i = 0; i < nodes; i++) {
        is_valid_nodes[i] = 1;
    }

    cudaMalloc(&d_col, sizeof(int) * edges);
    cudaMalloc(&d_rowdeg, sizeof(int) * nodes);
    cudaMalloc(&d_row, sizeof(int) * edges);
    cudaMalloc(&d_colptr, sizeof(int) * (nodes + 1));
    cudaMalloc(&num_active_nodes_d, sizeof(unsigned int));

    cudaMemcpy(d_value, value, sizeof(float) * nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_is_valid_nodes, is_valid_nodes, sizeof(int) * nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowdeg, rowdeg, sizeof(int) * nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, row, sizeof(int) * edges, cudaMemcpyHostToDevice);
    cudaMemcpy(d_colptr, colptr, sizeof(int) * (nodes + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col, sizeof(int) * edges, cudaMemcpyHostToDevice);
    cudaMemcpy(num_active_nodes_d, &num_active_nodes, sizeof(unsigned int), cudaMemcpyHostToDevice);

    while (true) {
        computeValue<<<nodes/threads_per_block+1,threads_per_block, threads_per_block*sizeof(float)>>>(nodes, d_value, d_new_value, d_rowdeg, d_colptr, d_row, d_col, d_is_valid_nodes);
        cudaDeviceSynchronize();
        updateActive<<<nodes/threads_per_block+1,threads_per_block>>>(nodes, d_value, d_new_value, d_is_valid_nodes, num_active_nodes_d, epsilon);
        cudaDeviceSynchronize();
        cudaMemcpy(d_value, d_new_value, sizeof(float) * nodes, cudaMemcpyDeviceToDevice);
        
        cudaMemcpy(&num_active_nodes, num_active_nodes_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        //printf("%d\n", num_active_nodes_d);
        if (num_active_nodes == 0)
            break;
    }

    cudaMemcpy(value, d_value, sizeof(float) * nodes, cudaMemcpyDeviceToHost);

    cudaFree(d_value);
    cudaFree(d_new_value);

    cudaFree(d_is_valid_nodes);
    free(is_valid_nodes);

    cudaFree(d_rowdeg);
    cudaFree(d_colptr);
    cudaFree(d_row);
    cudaFree(d_col);
}

int main(int argc, char* argv[])
{   
    if (argc != 9 || strcasecmp("-f", argv[1]) || strcasecmp("-o", argv[3]) || strcasecmp("-e", argv[5]) || strcasecmp("-t", argv[7])) {
        fprintf(stderr, "Usage ./pagerank -f (file name) -o (output file name) -e (epsilon) -t (thread per block)\n");
        return 1;
    }

    char *filename = argv[2];
    char *output = argv[4];
    double epsilon = atof(argv[6]);
    int threads_per_block = atof(argv[8]);
    int nodes;
    int edges;

    clock_t start, end;
    FILE* file = fopen(filename, "r");
    assert(file != NULL);

    fscanf(file, "%d %d\n", &nodes, &edges);

    float* value = (float*)malloc(nodes * sizeof(float));
    int* rowdeg = (int*)malloc(nodes * sizeof(int));
    int* colptr = (int*)malloc((nodes + 1) * sizeof(int));
    int* row = (int*)malloc(edges * sizeof(int));
    int* col = (int*)malloc(edges * sizeof(int));

    //row is the from node, col is the to node
    int i = 0, j = 0;

    //row is the from node, col is the to node
    while(i < edges)
    {
        fscanf(file, "%d %d\n", &col[i], &row[i]);
        i++;
    }

    start = clock();
    i = 0;
    while(i < nodes) {
        rowdeg[i] = 0;
        value[i] = alpha;
        i++;
    }
    i = 0;
    colptr[0] = 0;
    while(i < edges) {
        rowdeg[row[i]]++;
        while (j < col[i]) {
            colptr[++j] = i;
        }
        i++;
    }
    colptr[nodes] = edges;
    

    fclose(file);

    pagerank(nodes, edges, value, rowdeg, colptr, row, col, epsilon, threads_per_block);

    end = clock();
    double time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("GPU async atomicAdd cost: %fs\n", time_taken);

    // write result to file
    FILE* fout = fopen(output, "w");
    assert(fout != NULL);
    for (int i = 0; i < nodes; i++) {
        fprintf(fout, "%d  %f\n", i, value[i]);
    }
    fclose(fout);

    free(value);
    free(rowdeg);
    free(colptr);
    free(row);
    free(col);


    return 0;
}
