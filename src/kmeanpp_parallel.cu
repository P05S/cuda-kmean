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




///---------------------------SCAN from lab to help with calculations---------------------------


__global__ void blockInclusiveScan(double* input, double *output, double* auxArr, int n){
    __shared__ double tempArr[2*BLOCK_SIZE];

    int tid = threadIdx.x;
    int start = 2 * blockIdx.x * BLOCK_SIZE;
    int idx = start + tid;

    if (idx < n){
        tempArr[tid] = input[idx];
    } else {
        tempArr[tid] = 0;
    }
    if (idx + BLOCK_SIZE < n){
        tempArr[tid + BLOCK_SIZE] = input[idx+BLOCK_SIZE];
    } else {
        tempArr[tid + BLOCK_SIZE] = 0;
    }
    __syncthreads();



    for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < 2 * BLOCK_SIZE) {
            tempArr[index] += tempArr[index - stride];
        }
        __syncthreads();
    }

    
    for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
        int index = (tid + 1) * 2 * stride - 1;

        if (index + stride < 2*BLOCK_SIZE) {
            tempArr[index + stride] += tempArr[index];
        }
        __syncthreads();
    }

    __syncthreads();
   
    if (idx < n){
        output[idx] = tempArr[tid];
    }
    if (idx + BLOCK_SIZE < n){
        output[idx + BLOCK_SIZE] = tempArr[tid + BLOCK_SIZE];
    }

    if (tid == 0){
        auxArr[blockIdx.x] = tempArr[2*BLOCK_SIZE - 1];
    }

}

__global__ void inclusiveAddBlockSum(double* blockSum, double* output, int n){
    if (blockIdx.x == 0) return;
    int start = 2 * blockIdx.x * BLOCK_SIZE;
    int idx = start + threadIdx.x;
    if (idx < n){
        output[idx] += blockSum[blockIdx.x-1];
    }
    if (idx + BLOCK_SIZE < n){
        output[idx + BLOCK_SIZE] += blockSum[blockIdx.x-1];
    }
}

void inclusiveScan(double* input, double* output, int n){
    int blocks = (n + 2* BLOCK_SIZE - 1) / (BLOCK_SIZE*2);

    double* auxArr;
    cudaMalloc( (void**) &auxArr, sizeof(double)*blocks);
    
    blockInclusiveScan<<<blocks, BLOCK_SIZE>>>(input, output, auxArr, n);
    if (blocks > 1){
        double *blockSum;
        cudaMalloc( (void**) &blockSum, sizeof(double)*blocks);

        inclusiveScan(auxArr, blockSum, blocks);

        inclusiveAddBlockSum<<<blocks, BLOCK_SIZE>>>(blockSum, output, n);
        
        cudaFree(blockSum);
    }
    cudaFree(auxArr);
}


//-------------------------------------------------------------------------------------------------

__global__ void compute_distance(double * data, double* centroids, int k, int rows, int cols, int* r, int current_iter,  double* distArr) {
 

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= rows) {
        return;
    }

    double distToNearestCentroid = INFINITY;
    for (int i = 0; i < current_iter; i++) {
        double dist = 0;
        for (int j = 0; j < cols; j++) {
            double val = data[idx * cols + j] - centroids[i * cols + j];
            dist += val * val;
        }
        if (dist < distToNearestCentroid) {
            distToNearestCentroid = dist;
        }
    }

    distArr[idx] = distToNearestCentroid;
}

__global__ void assign_clusters(double * data, double* centroids, int* cluster, double* centroids_change, int* centroids_count, int rows, int cols, int k, int* new_assignments) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) {
        return;
    }
    double minDist = INFINITY;
    int new_cluster = 0;
    for (int j = 0; j < k; j++) {
        double dist = 0;
        for (int i = 0; i < cols; i++) {
            double val = data[idx * cols + i] - centroids[j * cols + i];
            dist += val * val;
        }
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
    if (idx < k*cols) {
        int j = idx / cols;
        centroids[idx] = centroids_change[idx] / centroids_count[j];

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
        return NULL;
    }
    char line[MAX_LINE_LENGTH];
    int rows = 0;
    int cols = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (rows == 0)
            cols = count_columns(line);
        rows++;
    }

    if (rows == 0) { fclose(fp); return NULL; }
    rows -= 1; // remove header
    rewind(fp);

    // allocate result on heap
    int *result = (int*)malloc(3 * sizeof(int));
    if (!result) { fclose(fp); return NULL; }
    result[0] = 0; result[1] = rows; result[2] = cols;

    *data_ptr = (double**)malloc(rows * sizeof(double *));
    if (!*data_ptr) {
        perror("Memory allocation failed");
        fclose(fp);
        free(result);
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        (*data_ptr)[i] = (double*)malloc(cols * sizeof(double));
        if (!(*data_ptr)[i]) {
            perror("Memory allocation failed");
            // free previously allocated
            for (int t = 0; t < i; ++t) free((*data_ptr)[t]);
            free(*data_ptr);
            fclose(fp);
            free(result);
            return NULL;
        }
    }

    int row = 0;
    while (fgets(line, sizeof(line), fp)) {
        // skip header
        if (row == 0) { row++; continue; }
        line[strcspn(line, "\n")] = 0;
        char *token = strtok(line, ",");
        int col = 0;
        while (token != NULL && col < cols) {
            (*data_ptr)[row-1][col] = strtod(token, NULL);
            col++;
            token = strtok(NULL, ",");
        }
        row++;
        if (row > rows) break;
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
    double* centroids;
    int* new_assignments;
    int* r;
    double* data_flatten = (double*)malloc(rows * cols * sizeof(double));
    double* cumulative_dist = (double*)malloc(rows * sizeof(double));
    for (int i = 0; i < rows; i++) {
        memcpy(&data_flatten[i * cols], data[i], cols * sizeof(double));
    }

    cluster = (int*)malloc(rows * sizeof(int));
    new_assignments = (int*)malloc(sizeof(int));
    r = (int*)malloc(K*sizeof(int));

    centroids = (double*)malloc(K * cols * sizeof(double));


    double* device_data;
    double* device_centroids;
    double* device_centroids_change;
    int* device_r;
    int* device_centroids_count;
    int* device_cluster;
    int* device_new_assignments;
    double* distArr;
    double* device_cumulative_dist;
    cudaMalloc(&device_data, rows * cols * sizeof(double));
    cudaMalloc(&device_centroids, K * cols * sizeof(double));
    cudaMalloc(&device_centroids_change, K * cols * sizeof(double));
    cudaMalloc(&device_centroids_count, K * sizeof(int));
    cudaMalloc(&device_cluster, rows * sizeof(int));
    cudaMalloc(&device_new_assignments, sizeof(int));
    cudaMalloc(&device_r, K * sizeof(int));
    cudaMalloc(&distArr, rows * sizeof(double));
    cudaMalloc(&device_cumulative_dist, rows * sizeof(double));

    auto start_algo = std::chrono::high_resolution_clock::now();
    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventRecord(kernel_start);

    for (int i=0; i < K; i++) {
        r[i] = rand();
    }


    cudaMemcpy(device_data, data_flatten, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_r, r, K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(device_centroids_change, 0, K * cols * sizeof(double));
    cudaMemset(device_centroids_count, 0, K * sizeof(int));
    cudaMemset(device_cluster, -1, rows * sizeof(int));
    

    int blocks = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;


    cudaMemcpy(device_centroids, &data_flatten[int(r[0]%rows)],  cols * sizeof(double), cudaMemcpyHostToDevice);
    for (int i = 1; i < K; i++) {
        compute_distance<<<blocks, BLOCK_SIZE>>>(device_data, device_centroids, K, rows,cols, device_r, i, distArr);
        cudaDeviceSynchronize();
        inclusiveScan(distArr, device_cumulative_dist, rows);

        cudaMemcpy(cumulative_dist, device_cumulative_dist, rows * sizeof(double), cudaMemcpyDeviceToHost);
        double total = cumulative_dist[rows - 1];
        double scaled_r = (double)r[i] / (double)RAND_MAX * total;

        int left = 0;
        int right = rows - 1;
        int j = right;   
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (cumulative_dist[mid] > scaled_r) {
                j = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        cudaMemcpy(device_centroids + i*cols, data_flatten + j*cols, cols * sizeof(double), cudaMemcpyHostToDevice);
    }


    for (int iter = 0; iter < MAX_ITER; iter++) {
        cudaMemset(device_new_assignments, 0, sizeof(int));
        assign_clusters<<<blocks, BLOCK_SIZE>>>(device_data, device_centroids, device_cluster, device_centroids_change, device_centroids_count, rows, cols, K, device_new_assignments);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&new_assignments[0], device_new_assignments, sizeof(int), cudaMemcpyDeviceToHost);

        if (new_assignments[0] == 0) {
            printf("Converged in %d iterations\n", iter);
            break;
        }

        update_centroids<<<((K+cols) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(device_centroids, device_centroids_change, device_centroids_count, K, cols);
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
    cout << "Algorithm Time taken: " << elapsed.count() << " ms" << endl;

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
    cudaFree(device_new_assignments);
    cudaFree(device_r);
    cudaFree(distArr);
    cudaFree(device_cumulative_dist);

    
    write_clusters_csv("../output/clusters_kmeanpp_parallel.csv", cluster, rows);
    write_centroids_csv("../output/centroids_kmeanpp_parallel.csv", centroids, K, cols);
    free(new_assignments);
    free(data_flatten);
    free(r);
    free(cumulative_dist);
    free(cluster);
    free(result);
    for (int i = 0; i < rows; i++) {
        free(data[i]);
    }
    free(data);
    return 0;
}
