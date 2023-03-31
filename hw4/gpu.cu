#include "common.h"
#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <cstdio>
#include <cstdlib>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
static int blks;
static int numRows;
static int totalBins;
static int* myBin; // myBin[i] is the bin number ith particle in parts
static int* binCounts; // binCounts[i] is the number of particles in bin i
static int* binOffsets;  // binOffsets[i] is the offset of bin i in binIndices
static int* binIndices;  // binIndices track indices in sorted bin order

 
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

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
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

   // particle.ax += coef * dx;
    atomicAdd1(&particle.ax, coef * dx);
   // particle.ay += coef * dy;
    atomicAdd1(&particle.ay, coef * dy);
}

// void prefix_sum(int totalBins, int* binOffsets, int* binCounts){
//     thrust::inclusive_scan(binCounts, binCounts + totalBins, binOffsets);
// }

__global__ void calculate_bin_counts(particle_t* particles, int num_parts, int numRows, int* myBin, int* binCounts){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts){
        return;
    }

    particles[tid].ax = particles[tid].ay = 0;
    int col = particles[tid].x / cutoff;
    int row = particles[tid].y / cutoff;
    int bin = col + row * numRows;
    myBin[tid] = bin;
    atomicAdd(&binCounts[bin], 1);
    //prefix_sum(totalBins, binOffsets, binCounts);
}

__global__ void reshuffle(int num_parts, int* binIndices, int* binCounts, int* binOffsets, int* myBin, int totalBins){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts){
        return;
    }

    int bin = myBin[tid];
    atomicSub(&binCounts[bin], 1);
    binIndices[binOffsets[bin] + binCounts[bin]] = tid;
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int numRows, int* myBin, int* binOffsets, int* binIndices) {
    // Get thread (particle) ID

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts){
            return;
    }

    int col = particles[tid].x / cutoff;
    int row = particles[tid].y / cutoff;
    int bin = myBin[tid];

    bool hasLeft = col - 1 >= 0;
    bool hasRight = col + 1 < numRows;
    bool hasTop = row - 1 >= 0;
    bool hasBottom = row + 1 < numRows;

    // current
    for (int j = binOffsets[bin]; j < binOffsets[bin + 1]; ++j){
        apply_force_gpu(particles[tid], particles[binIndices[j]]);
    }

    // left
    if (hasLeft){
        for (int j = binOffsets[bin - 1]; j < binOffsets[bin]; ++j){
            apply_force_gpu(particles[tid], particles[binIndices[j]]);
        }
    }

    // right
    if (hasRight){
        for (int j = binOffsets[bin + 1]; j < binOffsets[bin + 2]; ++j){
            apply_force_gpu(particles[tid], particles[binIndices[j]]);
        }
    }

    if (hasTop){
        // current
        for (int j = binOffsets[bin - numRows]; j < binOffsets[bin - numRows + 1]; ++j){
            apply_force_gpu(particles[tid], particles[binIndices[j]]);
        }

        // left
        if (hasLeft){
            for (int j = binOffsets[bin - numRows - 1]; j < binOffsets[bin - numRows]; ++j){
                apply_force_gpu(particles[tid], particles[binIndices[j]]);
            }
        }

        // right
        if (hasRight){
            for (int j = binOffsets[bin - numRows + 1]; j < binOffsets[bin - numRows + 2]; ++j){
                apply_force_gpu(particles[tid], particles[binIndices[j]]);
            }
        }

    }

    if (hasBottom){
        // current
        for (int j = binOffsets[bin + numRows]; j < binOffsets[bin + numRows + 1]; ++j){
            apply_force_gpu(particles[tid], particles[binIndices[j]]);
        }

        // left
        if (hasLeft){
            for (int j = binOffsets[bin + numRows - 1]; j < binOffsets[bin + numRows]; ++j){
                apply_force_gpu(particles[tid], particles[binIndices[j]]);
            }
        }

        // right
        if (hasRight){
            for (int j = binOffsets[bin + numRows + 1]; j < binOffsets[bin + numRows + 2]; ++j){
                apply_force_gpu(particles[tid], particles[binIndices[j]]);
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
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

void init_simulation(particle_t* parts, int num_parts, double size) {

    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins

    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    numRows = (size / cutoff) + 1;
    totalBins = numRows * numRows;

    // allocate memory for the arrays
    cudaMalloc((void**) &myBin, num_parts * sizeof(int));
    cudaMalloc((void**) &binIndices, num_parts * sizeof(int));
    cudaMalloc((void**) &binCounts, totalBins * sizeof(int));
    cudaMalloc((void**) &binOffsets, (totalBins + 1) * sizeof(int));
}

__global__ void prefix_sum(int* binCounts, int* binOffsets, int totalBins) {
    for (int i = 0; i < totalBins; ++i){
        binOffsets[i + 1] = binOffsets[i] + binCounts[i];
    }
    
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // count number of particlces in each bin
    calculate_bin_counts<<<blks, NUM_THREADS>>>(parts, num_parts, numRows, myBin, binCounts);


    // update offset array base on bin counts
    //thrust::inclusive_scan(binCounts, binCounts + 2, binOffsets);
    thrust::device_ptr<int> binCounts_device(binCounts);
    thrust::device_ptr<int> binOffsets_device(binOffsets + 1);
    thrust::inclusive_scan(binCounts_device, binCounts_device + totalBins, binOffsets_device);

    // update bin indices array (sorting)
    reshuffle<<<blks, NUM_THREADS>>>(num_parts, binIndices, binCounts, binOffsets, myBin, totalBins);

    // compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, numRows, myBin, binOffsets, binIndices);

    // move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
