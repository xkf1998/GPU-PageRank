#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include <cuda.h>

const float alpha = 0.85;


__global__ void collectResults(const int nodes, float* value, float* message, const int* colptr) 
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < nodes) {
        float tmp = 1 - alpha;
        int e = colptr[tid];
        while (e < colptr[tid + 1]) {
            tmp += message[e];
            e++;
        }
        value[tid] = tmp;
    }
}


__global__ void computeValue(const int edges, float* value, float* message, const int* rowdeg, const int* row, const int* col) 
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < edges) {
        message[tid] = alpha * value[row[tid]] / (float)rowdeg[row[tid]];
    }
}

// message aggregation undone
void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* colptr, const int* row, const int* col, const int iteration, const int threads_per_block)
{
    int *d_rowdeg, *d_colptr, *d_row, *d_col;
    float *d_value, *d_message;

    cudaMalloc(&d_row, sizeof(int) * edges);
    cudaMalloc(&d_rowdeg, sizeof(int) * nodes);
    cudaMalloc(&d_colptr, sizeof(int) * (nodes + 1));
    cudaMalloc(&d_value, sizeof(float) * nodes);
    cudaMalloc(&d_message, sizeof(float) * edges);
    cudaMalloc(&d_col, sizeof(int) * edges);


    cudaMemcpy(d_colptr, colptr, sizeof(int) * (nodes + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, row, sizeof(int) * edges, cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, sizeof(float) * nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowdeg, rowdeg, sizeof(int) * nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col, sizeof(int) * edges, cudaMemcpyHostToDevice);

    int i = 0;
    while(i < iteration) {
        computeValue<<<edges/threads_per_block+1, threads_per_block>>>(edges, d_value, d_message, d_rowdeg, d_row, d_col);
        cudaDeviceSynchronize();
        collectResults<<<nodes/threads_per_block+1, threads_per_block>>>(nodes, d_value, d_message, d_colptr);
        cudaDeviceSynchronize();
        i++;
    }

    cudaMemcpy(value, d_value, sizeof(float) * nodes, cudaMemcpyDeviceToHost);

    cudaFree(d_value);
    cudaFree(d_rowdeg);
    cudaFree(d_colptr);
    cudaFree(d_row);
    cudaFree(d_col);
}

int main(int argc, char* argv[])
{   
    if (argc != 9 || strcasecmp("-f", argv[1]) || strcasecmp("-o", argv[3]) || strcasecmp("-i", argv[5]) || strcasecmp("-t", argv[7])) {
        fprintf(stderr, "Usage ./pagerank -f (file name) -o (output file name) -i (iterations) -t (thread per block)\n");
        return 1;
    }

    char *filename = argv[2];
    char *output = argv[4];
    int iteration = atoi(argv[6]);; // for synchronize update
    int threads_per_block = atoi(argv[8]);;
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

    pagerank(nodes, edges, value, rowdeg, colptr, row, col, iteration, threads_per_block);

    end = clock();
    double time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("GPU sync cost: %fs\n", time_taken);
    
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