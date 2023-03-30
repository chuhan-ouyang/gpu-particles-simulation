#include "common.h"
#include <cuda.h>
#include <thrust/scan.h>
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

// use thrust device ptr to copy the binCounts
// input: binCounts, output: binOffsets; both need to be device ptr

// need atomic structure here
// must use atomic 

// the only device 
__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// void prefix_sum(int totalBins, int* binOffsets, int* binCounts){
//     thrust::inclusive_scan(binCounts, binCounts + totalBins, binOffsets);
// }

// sorting the bin Indices; put indices of particles into this 'binIndcies' array such that it is sorted by binID
// loop through all particles, compute bin ID, 
// we have binCounts
// curent particle is in second bin, know the offset, just need to put that in the n + 1..

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

// prefix sum calculated
// bin Indices: sorted by bin id

// use atomic add in compute force
__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int numRows, int* myBin, int* binOffsets, int* binIndices) {
    // Get thread (particle) ID
    
    // need strides
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts){
            return;
    }
    
    // one thread takes care of one bin
    // first for loop looping over all particles in this bin
    // inner loop loop over all in the neighbor
    
    // when apply force, update forces in pairs
    // compute once but update the force for two particles at a time
    // when you do 9 neighbors, just compute the upper left, upper, upper right, left and itself
    
    // column and row of this current bin number 
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

__global__ void move_gpu(particle_t* particles, int num_parts, double& size) {
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

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS; // for large particle sizes, get numnb
    
    //cudaGetDevice(&devId); 
    //cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
    //numRows = (size / cutoff) + 1;
    //totalBins = numRows * numRows;
    //number of threads should be multiple of 32
    //number of blocks should be multiple of SMs (can try with different mulitple), need stride with number of threads

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
    // thrust::inclusive_scan(binCounts, binCounts + totalBins, binOffsets + 1);
    thrust::device_ptr<int> binCounts_devide(binCounts);
    thrust::device_ptr<int> binCounts_device(binCounts);
    thrust::inclusive_scan(binCounts, binCounts + totalBins, binOffsets);
    //prefix_sum<<<blks, NUM_THREADS>>>(binCounts_devide, binCounts_devide, binCounts_device);

    // update bin indices array (sorting)
    reshuffle<<<blks, NUM_THREADS>>>(num_parts, binIndices, binCounts, binOffsets, myBin, totalBins);

    // compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, numRows, myBin, binOffsets, binIndices);

    // move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
