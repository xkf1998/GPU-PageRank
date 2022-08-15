#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
// #include <omp.h>

// const int threads_per_block = 512;
const float alpha = 0.85;

void pagerank(const int nodes, const int edges, float *value, const int *rowdeg, const int *colptr, const int *row, const int *col, const double epsilon)
{
    float *new_value = new float[nodes];
    int *is_valid_node = new int[nodes];
    int num_active_nodes;

    for (int n = 0; n < nodes; n++)
    {
        is_valid_node[n] = 1;
    }

    while (true)
    {
        for (int n = 0; n < nodes; n++)
        {
            if (is_valid_node[n] != 1)
                continue;
            new_value[n] = 1 - alpha;
            for (int e = colptr[n]; e < colptr[n + 1]; e++)
            {
                new_value[n] += alpha * value[row[e]] / (float)rowdeg[row[e]];
            }
        }

        for (int n = 0; n < nodes; n++)
        {
            if (is_valid_node[n] != 1)
                continue;
            if (abs(value[n] - new_value[n]) < epsilon)
            {
                is_valid_node[n] = 0;
            }
        }

        for (int n = 0; n < nodes; n++)
        {
            value[n] = new_value[n];
        }
        num_active_nodes = 0;
        for (int i = 0; i < nodes; i++)
        {
            if (is_valid_node[i] == 1)
                num_active_nodes++;
        }
        if (num_active_nodes == 0)
            break;
    }

    delete[] new_value;
    delete[] is_valid_node;
}

int main(int argc, char *argv[])
{
    if (argc != 7 || strcasecmp("-f", argv[1]) || strcasecmp("-o", argv[3]) || strcasecmp("-e", argv[5]))
    {
        fprintf(stderr, "Usage ./pagerank -f (file name) -o (output file name) -e (epsilon)\n");
        return 1;
    }

    char *filename = argv[2];
    char *output_filename = argv[4];
    double epsilon = atof(argv[6]);
    int nodes;
    int edges;

    clock_t start, end;

    FILE *file = fopen(filename, "r");
    assert(file != NULL);

    fscanf(file, "%d %d\n", &nodes, &edges);

    float *value = (float *)malloc(nodes * sizeof(float));
    int *rowdeg = (int *)malloc(nodes * sizeof(int));
    int *colptr = (int *)malloc((nodes + 1) * sizeof(int));
    int *row = (int *)malloc(edges * sizeof(int));
    int *col = (int *)malloc(edges * sizeof(int));

    int i = 0, j = 0;

    // row is the from node, col is the to node
    while (i < edges)
    {
        fscanf(file, "%d %d\n", &col[i], &row[i]);
        i++;
    }

    start = clock();
    i = 0;
    while (i < nodes)
    {
        rowdeg[i] = 0;
        value[i] = alpha;
        i++;
    }

    i = 0;
    colptr[0] = 0;
    while (i < edges)
    {
        rowdeg[row[i]]++;
        while (j < col[i])
        {
            colptr[++j] = i;
        }
        i++;
    }
    colptr[nodes] = edges;
    fclose(file);

    pagerank(nodes, edges, value, rowdeg, colptr, row, col, epsilon);

    end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CPU serial cost: %lfs\n", time_taken);

    // write result to file
    FILE *fout = fopen(output_filename, "w");
    assert(fout != NULL);
    for (int i = 0; i < nodes; i++)
    {
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
