#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>


const float alpha = 0.85;
// __device__ __managed__ double epsilon = 0.00015; // for asynchronize update


// message aggregation
__global__ void compute(const int nodes, float* value, float* new_value, const int* rowdeg, const int* colptr, const int* row, const int* col, int* is_valid_nodes)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ float tile[];
    tile[threadIdx.x] = 1 - alpha;
    if (tid < nodes && is_valid_nodes[tid] == 1) {

        for (int e = colptr[tid]; e < colptr[tid + 1]; e++) {
            tile[threadIdx.x] += alpha * value[row[e]] / (float)rowdeg[row[e]];
        }

        new_value[tid] = tile[threadIdx.x];
    }
}

__global__ void find_active(const int nodes, float* __restrict__ value, float* __restrict__ new_value, int* __restrict__ is_valid_nodes, const double epsilon)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nodes && is_valid_nodes[tid] == 1) {
        is_valid_nodes[tid] = abs(value[tid] - new_value[tid]) > epsilon ? 1 : 0;
    }
}

__global__ void init_is_valid_nodes(int* __restrict__ is_valid_nodes, const int nodes) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < nodes)
        is_valid_nodes[tid] = 1;
}

void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* colptr, const int* row, const int* col, const double epsilon, const int threads_per_block)
{
    float *new_value;
    int *is_valid_nodes;
    cudaMallocManaged(&new_value, sizeof(float) * nodes);
    cudaMallocManaged(&is_valid_nodes, sizeof(int) * nodes);
    
    // init is_valid_nodes
    init_is_valid_nodes<<<nodes/threads_per_block+1,threads_per_block>>>(is_valid_nodes, nodes);
    while (true) {
        compute<<<nodes/threads_per_block+1, threads_per_block, threads_per_block*sizeof(float)>>>(nodes, value, new_value, rowdeg, colptr, row, col, is_valid_nodes);
        cudaDeviceSynchronize();
        find_active<<<nodes/threads_per_block+1, threads_per_block>>>(nodes, value, new_value, is_valid_nodes, epsilon);
        cudaDeviceSynchronize();
        memcpy(value, new_value, sizeof(float) * nodes);
        int num_active_nodes = 0;
        for (int i = 0; i < nodes; i++) {
            if (is_valid_nodes[i] == 1) {
                num_active_nodes++;
            }
        }

        //calculate how many active nodes;
        if (num_active_nodes == 0)
            break;
    }

    cudaFree(new_value);
    cudaFree(is_valid_nodes);
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

    float* value;
    int* rowdeg;
    int* colptr;
    int* row;
    int* col;

    cudaMallocManaged(&value, sizeof(float) * nodes);
    cudaMallocManaged(&rowdeg, sizeof(int) * nodes);
    cudaMallocManaged(&colptr, sizeof(int) * (nodes + 1));
    cudaMallocManaged(&row, sizeof(int) * edges);
    cudaMallocManaged(&col, sizeof(int) * edges);

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

    colptr[0] = 0;
    i = 0;
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
    printf("GPU async cost: %fs\n", time_taken);

    // write result to file
    FILE* fout = fopen(output, "w");
    assert(fout != NULL);
    for (int i = 0; i < nodes; i++) {
        fprintf(fout, "%d  %f\n", i, value[i]);
    }
    fclose(fout);

    cudaFree(value);
    cudaFree(rowdeg);
    cudaFree(colptr);
    cudaFree(row);
    cudaFree(col);


    return 0;
}