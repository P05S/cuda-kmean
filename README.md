# 🚀 CUDA K-MEANS

This project implements a high-performance CUDA-accelerated K-Means clustering algorithm 

---

## 🧩 Features

- GPU-Accelerated K-Means++
- K-Means++ Initialization
- Shared memory optimization
- CSV Export

---

## 📁 Project Structure
cuda-clahe/
│
├── src/
│ ├── kmean_parallel
│ ├── kmean_sequential
│ ├── kmeanpp_parallel
│ └── kmeanpp_shared
│
├── data/
│ └──  boxes3.csv
│
├── output/
│ ├── centroids_kmean_sequential.csv
│ ├── centroids_kmean_parallel.csv
│ ├── centroids_kmeanpp_parallel.csv
│ ├── centroids_kmeanpp_shared.csv
│ ├── clusters_kmean_sequential.csv
│ ├── clusters_kmean_parallel.csv
│ ├── clusters_kmeanpp_parallel.csv
│ └── clusters_kmeanpp_shared.csv
│
├── output_visualization.ipynb
├──README.md # Project documentation
└── presentation.pdf # slide

---

## ⚙️ Requirements
- **CUDA Toolkit** 
- **NVIDIA GPU with CUDA capability (Compute Capability ≥ 6.0)**


---

## 🔧 Build Instructions

**Make sure:**

- You have CUDA Toolkit installed

- Your compiler (nvcc) works from terminal

Then compile cuda file using the command like the following:


    nvcc -arch=sm_86 kmean_parallel.cu -o kmean.exe



💡 Note:

- If your GPU has a different compute capability, replace sm_86 with the appropriate value.
You can find your GPU's architecture at: https://developer.nvidia.com/cuda-gpus
---

## ▶️ **Run the Program**

    ./kmeans.exe

By default, it reads:

    ../data/boxes3.csv

And outputs results to:

    ../output/clusters_kmean_parallel.csv
    ../output/centroids_kmean_parallel.csv

## Presentation link

https://www.canva.com/design/DAG5Sqy4nKA/pgD8Vzvrisuj-s9A0Zp_5g/edit?utm_content=DAG5Sqy4nKA&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

