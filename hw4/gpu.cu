#include "common.h"
#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <cstdio>
#include <cstdlib>

#define NUM_THREADS 256
#define getBin(p, numRows, cutoff) (int)(p.x/cutoff) + (int)(p.y/cutoff) * numRows

// Put any static global variables here that you will use throughout the simulation.
static int blks;
static int numRows;
static int totalBins;
static int* myBin; // myBin[i] is the bin number of the ith particle in parts
static int* binCounts; // binCounts[i] is the number of particles in bin i
static int* binOffsets;  // binOffsets[i] is the offset of bin i in binIndices
static int* binIndices;  // binIndices track indices of particles sorted in bin order

static int devId;
static int numSMs;
 
__device__ double atomicAdd1(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
 
 __device__ void apply_force_gpu_pairs(particle_t& particle, particle_t& neighbor) {

    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;

    atomicAdd1(&particle.ax, coef * dx);
    atomicAdd1(&particle.ay, coef * dy);
    atomicAdd1(&neighbor.ax, -coef * dx);
    atomicAdd1(&neighbor.ay, -coef * dy);
}


__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {

    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;

    atomicAdd1(&particle.ax, coef * dx);
    atomicAdd1(&particle.ay, coef * dy);
}


__global__ void calculate_bin_counts(particle_t* particles, int num_parts, int numRows, int* myBin, int* binCounts){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_parts; i += stride){
        particles[i].ax = particles[i].ay = 0;
        myBin[i] = getBin(particles[i], numRows, cutoff);
        atomicAdd(&binCounts[myBin[i]], 1);
    }
}

__global__ void reshuffle(int num_parts, int* binIndices, int* binCounts, int* binOffsets, int* myBin, int totalBins){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_parts; i += stride){
        int bin = myBin[i];
        int position = atomicSub(&binCounts[bin], 1);
        binIndices[binOffsets[bin] + position - 1] = i;
    }
}


__global__ void compute_forces_gpu_new(particle_t* particles, int num_parts, int numRows, int* myBin, int* binOffsets, int* binIndices, int totalBins) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int dirs[5][2] = {{0,0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

    for (int i = index; i < totalBins; i += stride){
        int r = i / numRows;
        int c = i % numRows;

        for (int j = 0; j < 5; j++){
            int newRow = r + dirs[j][0];
            int newCol = c + dirs[j][1];

            if (r == newRow && c == newCol){
                for (int k = binOffsets[i]; k < binOffsets[i + 1]; ++k){
                    for (int l = k + 1; l < binOffsets[i + 1]; ++l){
                        apply_force_gpu_pairs(particles[binIndices[k]], particles[binIndices[l]]);
                    }
                }
            }
            
            else if (0 <= newRow && newRow < numRows && 0 <= newCol && newCol < numRows){
                int bin = newRow * numRows + newCol;
                for (int k = binOffsets[i]; k < binOffsets[i + 1]; ++k){
                    for (int l = binOffsets[bin]; l < binOffsets[bin + 1]; ++l){
                        apply_force_gpu_pairs(particles[binIndices[k]], particles[binIndices[l]]);
                    }
                }
            }
        }
    }
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int numRows, int* myBin, int* binOffsets, int* binIndices, int totalBins) {
    // Get thread (particle) ID

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_parts; i += stride){
        int col = particles[i].x / cutoff;
        int row = particles[i].y / cutoff;
        int bin = myBin[i];

        bool hasLeft = col - 1 >= 0;
        bool hasRight = col + 1 < numRows;
        bool hasBottom = row + 1 < numRows;

        for (int j = binOffsets[bin]; j < binOffsets[bin + 1]; ++j){
            if (i < binIndices[j]){
                apply_force_gpu_pairs(particles[i], particles[binIndices[j]]);
            }
        }
    
        if (hasRight){
            for (int j = binOffsets[bin + 1]; j < binOffsets[bin + 2]; ++j){
                apply_force_gpu_pairs(particles[i], particles[binIndices[j]]);
            }
        }

        if (hasBottom){
            // current
            for (int j = binOffsets[bin + numRows]; j < binOffsets[bin + numRows + 1]; ++j){
                apply_force_gpu_pairs(particles[i], particles[binIndices[j]]);
            }

            // left
            if (hasLeft){
                for (int j = binOffsets[bin + numRows - 1]; j < binOffsets[bin + numRows]; ++j){
                    apply_force_gpu_pairs(particles[i], particles[binIndices[j]]);
                }
            }

            // right
            if (hasRight){
                for (int j = binOffsets[bin + numRows + 1]; j < binOffsets[bin + numRows + 2]; ++j){
                    apply_force_gpu_pairs(particles[i], particles[binIndices[j]]);
                }
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    // Get thread (particle) ID
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_parts; i += stride){
        particle_t* p = &particles[i];
        p->vx += p->ax * dt;
        p->vy += p->ay * dt;
        p->x += p->vx * dt;
        p->y += p->vy * dt;

        while (p->x < 0 || p->x > size) {
            p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
            p->vx = -(p->vx);
        }
        while (p->y < 0 || p->y > size) {
            p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
            p->vy = -(p->vy);
        }
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Do not do any particle simulation here

    cudaGetDevice(&devId); 
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);

    blks = 32 * numSMs;
    numRows = (size / cutoff) + 1;
    totalBins = numRows * numRows;

    // allocate memory for the arrays
    cudaMalloc((void**) &myBin, num_parts * sizeof(int));
    cudaMalloc((void**) &binIndices, num_parts * sizeof(int));
    cudaMalloc((void**) &binCounts, totalBins * sizeof(int));
    cudaMalloc((void**) &binOffsets, (totalBins + 1) * sizeof(int));
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // count number of particlces in each bin
    calculate_bin_counts<<<blks, NUM_THREADS>>>(parts, num_parts, numRows, myBin, binCounts);

    // prefix sum on binCounts to compute binOffsets
    thrust::device_ptr<int> binCounts_device(binCounts);
    thrust::device_ptr<int> binOffsets_device(binOffsets + 1);
    thrust::inclusive_scan(binCounts_device, binCounts_device + totalBins, binOffsets_device);

    // update bin indices array (sorting)
    reshuffle<<<blks, NUM_THREADS>>>(num_parts, binIndices, binCounts, binOffsets, myBin, totalBins);

    // compute forces
    compute_forces_gpu_new<<<blks, NUM_THREADS>>>(parts, num_parts, numRows, myBin, binOffsets, binIndices, totalBins);

    // move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
