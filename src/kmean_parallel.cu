#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip> 

using namespace std;


#define K 12     
#define MAX_ITER 1000

#define MAX_LINE_LENGTH 1024
#define BLOCK_SIZE 1024
#define CUDA_CHECK(msg) \
{ cudaError_t e = cudaGetLastError(); \
  if (e != cudaSuccess) { \
    std::cerr << "CUDA error after " << msg << ": " << cudaGetErrorString(e) << std::endl; \
    return -1; \
  } \
  e = cudaDeviceSynchronize(); \
  if (e != cudaSuccess) { \
    std::cerr << "CUDA sync error after " << msg << ": " << cudaGetErrorString(e) << std::endl; \
    return -1; \
  } \
}


__device__ double distance(double* data, double* centroids, int cols, int idx, int centroid_idx) {
    double sum = 0.0;
    for (int i = 0; i < cols; i++) {
        double val = data[idx * cols + i] - centroids[centroid_idx * cols + i];
        sum += val * val;
    }
    return sum;
}


__global__ void init_centroids(double * data, double* centroids, int k, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        for (int j = 0; j < cols; j++) {
            centroids[idx * cols + j] = data[idx * cols + j];
        }
    }
}

__global__ void assign_clusters(double * data, double* centroids, int* cluster, double* centroids_change, int* centroids_count, int rows, int cols, int k, int* new_assignments) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) {
        return;
    }
    double minDist = distance(data, centroids, cols, idx, 0);
    int new_cluster = 0;
    for (int j = 1; j < k; j++) {
        double dist = distance(data, centroids, cols, idx, j);
        if (dist < minDist) {
            minDist = dist;
            new_cluster = j;
        }
    }

    if (cluster[idx] != new_cluster) {
        new_assignments[0] = 1;
        for (int j = 0; j < cols; j++) {
            atomicAdd(&centroids_change[new_cluster * cols + j], data[idx * cols + j]);
            if (cluster[idx] != -1) {
                atomicAdd(&centroids_change[cluster[idx] * cols + j], -data[idx * cols + j]);
            }
        }
        atomicAdd(&centroids_count[new_cluster], 1);
        if (cluster[idx] != -1) {
            atomicAdd(&centroids_count[cluster[idx]], -1);
        }
        cluster[idx] = new_cluster;
    }
}

__global__ void update_centroids(double* centroids, double* centroids_change, int* centroids_count, int k, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        for (int j = 0; j < cols; j++) {
            centroids[idx * cols + j] = centroids_change[idx * cols + j] / centroids_count[idx];
        }
    }
}

// from GPT cuz I'm lazy
int count_columns(const char *line) {
    int count = 1;
    for (int i = 0; line[i]; i++) {
        if (line[i] == ',') count++;
    }
    return count;
}

int* loadCSV(char *filename, double ***data_ptr) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("Unable to open file %s\n", filename);
        return 0;
    }
    char line[MAX_LINE_LENGTH];
    int rows = 0;
    int cols = 0;
    int result[3] = {0, rows, cols};

    while (fgets(line, sizeof(line), fp)) {
        if (rows == 0)
            cols = count_columns(line);
        rows++;
    }

    rows -= 1; // remove header

    rewind(fp);
    result[1] = rows;
    result[2] = cols;
    *data_ptr = (double**)malloc((rows) * sizeof(double *));
    if (!*data_ptr) {
        perror("Memory allocation failed");
        fclose(fp);
        return result;
    }

    for (int i = 0; i < rows; i++) {
        (*data_ptr)[i] = (double*)malloc((cols) * sizeof(double));
        if (!(*data_ptr)[i]) {
            perror("Memory allocation failed");
            fclose(fp);
            return result;
        }
    }

    int row = 0;
    while (fgets(line, sizeof(line), fp) && row < rows+1) {
        line[strcspn(line, "\n")] = 0;  // remove newline

        char *token = strtok(line, ",");
        int col = 0;
        while (token != NULL && col < cols) {
            if (row == 0) {
                col++;
                token = strtok(NULL, ",");
                continue;
            }
            (*data_ptr)[row-1][col] = (double)strtod(token, NULL);  // convert to integer and store in token
            col++;
            token = strtok(NULL, ",");
        }
        row++;
    }

    fclose(fp);

    return result;
}

void write_clusters_csv(const std::string &filename,
                        const int *cluster,
                        int rows)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: cannot open " << filename << std::endl;
        return;
    }

    file << "row,cluster\n";
    for (int i = 0; i < rows; i++) {
        file << i << "," << cluster[i] << "\n";
    }
    file.close();
}


// Writes K centroids: one row per centroid: c0_0,c0_1,...,c0_cols
void write_centroids_csv(const std::string &filename,
                         const double *centroids,
                         int k,
                         int cols)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: cannot open " << filename << std::endl;
        return;
    }

    // header
    for (int j = 0; j < cols; j++) {
        file << "dim" << j;
        if (j + 1 < cols) file << ",";
    }
    file << "\n";

    // each centroid
    file << std::setprecision(10);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < cols; j++) {
            file << centroids[i * cols + j];
            if (j + 1 < cols) file << ",";
        }
        file << "\n";
    }

    file.close();
}


int main() {

    char *filename = "../data/boxes3.csv";
    double** data;
    int* result = loadCSV(filename, &data);
    int rows = result[1];
    int cols = result[2];

    auto start_end_to_end = std::chrono::high_resolution_clock::now();

    int* cluster;
    int* new_assignments;
    double* centroids;
    double* data_flatten = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++) {
        memcpy(&data_flatten[i * cols], data[i], cols * sizeof(double));
    }

    cluster = (int*)malloc(rows * sizeof(int));
    new_assignments = (int*)malloc(sizeof(int));
    centroids = (double*)malloc(K * cols * sizeof(double));

    double* device_data;
    double* device_centroids;
    double* device_centroids_change;
    int* device_centroids_count;
    int* device_cluster;
    int* device_new_assignments;
    cudaMalloc(&device_data, rows * cols * sizeof(double));
    cudaMalloc(&device_centroids, K * cols * sizeof(double));
    cudaMalloc(&device_centroids_change, K * cols * sizeof(double));
    cudaMalloc(&device_centroids_count, K * sizeof(int));
    cudaMalloc(&device_cluster, rows * sizeof(int));
    cudaMalloc(&device_new_assignments, sizeof(int));

    auto start_algo = std::chrono::high_resolution_clock::now();
    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventRecord(kernel_start);


    cudaMemcpy(device_data, data_flatten, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(device_centroids_change, 0, K * cols * sizeof(double));
    cudaMemset(device_centroids_count, 0, K * sizeof(int));
    cudaMemset(device_cluster, -1, rows * sizeof(int));

    init_centroids<<<((K + 128 - 1) / 128), 128>>>(device_data, device_centroids, K, cols);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        cudaMemset(device_new_assignments, 0, sizeof(int));
        int blocks = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        assign_clusters<<<blocks, BLOCK_SIZE>>>(device_data, device_centroids, device_cluster, device_centroids_change, device_centroids_count, rows, cols, K, device_new_assignments);
        // CUDA_CHECK("assign_clusters kernel");
        cudaDeviceSynchronize();
        cudaMemcpy(&new_assignments[0], device_new_assignments, sizeof(int), cudaMemcpyDeviceToHost);
        if (new_assignments[0] <= 0) {
            printf("Converged in %d iterations\n", iter);
            break;
        }

        blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
        update_centroids<<<blocks, BLOCK_SIZE>>>(device_centroids, device_centroids_change, device_centroids_count, K, cols);
        // CUDA_CHECK("update_centroids kernel");
        cudaDeviceSynchronize();
    }

    cudaMemcpy(cluster, device_cluster, rows * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids, device_centroids, K * cols * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);
    auto end = std::chrono::high_resolution_clock::now();

    float ms = 0;
    cudaEventElapsedTime(&ms, kernel_start, kernel_stop);
    printf("Kernel time: %.3f ms\n", ms);

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_algo);
    cout << "Algorithm time taken: " << elapsed.count() << " ms" << endl;

    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_end_to_end);
    cout << "Overall Time taken (Memory Allocation + Algo): " << elapsed.count() << " ms" << endl;

    for (int i = 0; i < 1000; i++) {
        printf("%d ", cluster[i]);
    }

    cudaFree(device_data);
    cudaFree(device_centroids);
    cudaFree(device_centroids_change);
    cudaFree(device_centroids_count);
    cudaFree(device_cluster);
    cudaFree(new_assignments);

    

    write_clusters_csv("../output/clusters_kmean_parallel.csv", cluster, rows);
    write_centroids_csv("../output/centroids_kmean_parallel.csv", centroids, K, cols);
    free(new_assignments);
    free(data_flatten);
    free(cluster);
    free(result);
    for (int i = 0; i < rows; i++) {
        free(data[i]);
    }
    free(data);
    return 0;
}
