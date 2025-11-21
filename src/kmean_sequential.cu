#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip> 

using namespace std;


#define K 12     
#define MAX_ITER 1000

#define MAX_LINE_LENGTH 1024


double distance(double* a, double* b, int cols) {
    double sum = 0.0;
    for (int i = 0; i < cols; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}


void init_centroids(double *** data_ptr, double*** centroids_ptr, int rows, int cols) {
    int random_idx = rand() % rows;
    memcpy((*centroids_ptr)[0], (*data_ptr)[random_idx], cols * sizeof(double));

    double* distArr;
    distArr = (double*)malloc(rows * sizeof(double));

    for (int i = 1; i < K; i++) {
        double total = 0.0;
        for (int j = 0; j < rows; j++) {
            double distToNearestCentroid  = distance((*data_ptr)[j], (*centroids_ptr)[0], cols);
            for (int k = 1; k < i; k++) {
                double dist= distance((*data_ptr)[j], (*centroids_ptr)[k], cols);
                if (dist < distToNearestCentroid) {
                    distToNearestCentroid = dist;
                }
            }
            distArr[j] = distToNearestCentroid;
            total += distToNearestCentroid;
        }

        double r = (double)rand() / (double)RAND_MAX * total;

        double cumul = 0.0;
        int chosen = 0;
        for (int j = 0; j < rows; j++) {
            cumul += distArr[j];
            if (cumul > r) {
                chosen = j;
                break;
            }
        }
        memcpy ((*centroids_ptr)[i], (*data_ptr)[chosen], cols * sizeof(double));
    }
}

void assign_clusters(double *** data_ptr, double*** centroids_ptr, int** cluster_ptr, int rows, int cols, int* new_assignments) {
    for (int i = 0; i < rows; i++) {
        int old_cluster = (*cluster_ptr)[i];
        double minDist = distance((*data_ptr)[i], (*centroids_ptr)[0], cols);
        int cluster = 0;
        for (int j = 1; j < K; j++) {
            double dist = distance((*data_ptr)[i], (*centroids_ptr)[j], cols);
            if (dist < minDist) {
                minDist = dist;
                cluster = j;
            }
        }

        if (old_cluster != cluster) {
            new_assignments[0] = 1;
        }
        (*cluster_ptr)[i] = cluster;
    }
}

void update_centroids(double *** data_ptr, double*** centroids_ptr, int** cluster_ptr, int rows, int cols) {
    int* count;

    count = (int*)malloc(K * sizeof(int));

    for (int i = 0; i < K; i++) {
        count[i] = 0;
        for (int j = 0; j < cols; j++) {
            (*centroids_ptr)[i][j] = 0.0;
        }
        
    }

    for (int i = 0; i < rows; i++) {
        int c = (*cluster_ptr)[i];
        for (int j = 0; j < cols; j++) {
            double val = (*data_ptr)[i][j];
            (*centroids_ptr)[c][j] += val;
        }
        count[c]++;
    }


    for (int i = 0; i < K; i++) {
        if (count[i] > 0) {
            for (int j = 0; j < cols; j++) {
                (*centroids_ptr)[i][j] /= count[i];
            }

        }
    }

    free(count);
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

    double** centroids;
    double** old_centroids;

    centroids = (double**)malloc(K * sizeof(double*));

    for (int i = 0; i < K; i++) {
        centroids[i] = (double*)malloc(cols * sizeof(double));
    }

    int* cluster;

    cluster = (int*)malloc(rows * sizeof(int));


    auto start_algo = std::chrono::high_resolution_clock::now();
    init_centroids(&data, &centroids , rows, cols);


    for (int iter = 0; iter < MAX_ITER; iter++) {
        int new_assignments = 0;
        assign_clusters(&data, &centroids , &cluster, rows, cols, &new_assignments);
        update_centroids(&data, &centroids , &cluster, rows, cols);

        if (new_assignments <= 0) {
            printf("Converged in %d iterations\n", iter + 1);
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_algo);
    cout << "Algorithm Time taken: " << elapsed.count() << " ms" << endl;

    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_end_to_end);
    cout << "Overall Time taken (Memory Allocation + Algo): " << elapsed.count() << " ms" << endl;

    for (int i = 0; i < 1000; i++) {
        printf("%d ", cluster[i]);
    }

    double* centroids_flattened = (double*)malloc(K * cols * sizeof(double));
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < cols; j++) {
            centroids_flattened[i * cols + j] = centroids[i][j];
        }
    }
    
    write_clusters_csv("../output/clusters_kmean_sequential.csv", cluster, rows);
    write_centroids_csv("../output/centroids_kmean_sequential.csv", centroids_flattened, K, cols);

    free(result);
    free(centroids_flattened);
    for (int i = 0; i < K; i++) {
        free(centroids[i]);
    }
    free(centroids);
    for (int i = 0; i < rows; i++) {
        free(data[i]);
    }
    free(data);
    return 0;
}
