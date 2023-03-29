#include "common.h"
#include <cmath>
#include <vector>
#include <iostream>

using namespace std;

int numRows;
int totalBins;
int* myBin; // myBin[i] is the bin number for the ith particle in parts
int* binCounts; // binCounts[i] is the number of particles in the ith bin
int* binOffsets; // each bin hold binOffsets[i]...binOffsets[i + 1] particles (inclusive left, exclusive rights)
// in the binIndices array
int* binIndices; // indices of particles in the parts array, sorted by bin ids
// in binIndices array that we can insert element into

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}


void init_simulation(particle_t* parts, int num_parts, double size) {
    // calcualte number of bins 
    numRows = (size / cutoff) + 1;
    totalBins = numRows * numRows;

    // allocate memory for the arrays
    myBin = new int[num_parts];
    binIndices = new int[num_parts];  

    binCounts = new int[totalBins];  
    binOffsets = new int[totalBins + 1];  
}


void simulate_one_step(particle_t* parts, int num_parts, double size) {
    
    // REBINNING

    // for each particle, calculate its bin number, add it to bin counts
    for (int i = 0; i < num_parts; ++i){
        parts[i].ax = parts[i].ay = 0;
        int col = parts[i].x / cutoff;
        int row = parts[i].y / cutoff;
        int bin = col + row * numRows;
        myBin[i] = bin;
        binCounts[bin]++;
    }

    // prefix sum
    binOffsets[0] = 0;
    for (int i = 0; i < totalBins; ++i){
        binOffsets[i + 1] = binOffsets[i] + binCounts[i];
    }
    

    // move particles to the correct bin_indices, updating updateCounter
    for (int i = 0; i < num_parts; ++i){
        int bin = myBin[i];
        binCounts[bin]--;
        binIndices[binOffsets[bin] + binCounts[bin]] = i;
    }

    // APPLY FORCE
    // apply force, only to those in the correct neighboring bins
    for (int i = 0; i < num_parts; ++i){
        int col = parts[i].x / cutoff;
        int row = parts[i].y / cutoff;
        int bin = myBin[i];

        bool hasLeft = col - 1 >= 0;
        bool hasRight = col + 1 < numRows;
        bool hasTop = row - 1 >= 0;
        bool hasBottom = row + 1 < numRows;

        // current
        // for all in this neighboring bin
        for (int j = binOffsets[bin]; j < binOffsets[bin + 1]; ++j){
            apply_force(parts[i], parts[binIndices[j]]);
        }

        // left
        if (hasLeft){
            // for all in this neighboring bin
            for (int j = binOffsets[bin - 1]; j < binOffsets[bin]; ++j){
                apply_force(parts[i], parts[binIndices[j]]);
            }
        }

        // right
        if (hasRight){
            // for all in this neighboring bin
            for (int j = binOffsets[bin + 1]; j < binOffsets[bin + 2]; ++j){
                apply_force(parts[i], parts[binIndices[j]]);
            }
        }

        if (hasTop){
            // current
            for (int j = binOffsets[bin - numRows]; j < binOffsets[bin - numRows + 1]; ++j){
                apply_force(parts[i], parts[binIndices[j]]);
            }

            // left
            if (hasLeft){
                for (int j = binOffsets[bin - numRows - 1]; j < binOffsets[bin - numRows]; ++j){
                    apply_force(parts[i], parts[binIndices[j]]);
                }
            }

            // right
            if (hasRight){
                for (int j = binOffsets[bin - numRows + 1]; j < binOffsets[bin - numRows + 2]; ++j){
                    apply_force(parts[i], parts[binIndices[j]]);
                }
            }

        }

        if (hasBottom){
            // current
            for (int j = binOffsets[bin + numRows]; j < binOffsets[bin + numRows + 1]; ++j){
                apply_force(parts[i], parts[binIndices[j]]);
            }

            // left
            if (hasLeft){
                for (int j = binOffsets[bin + numRows - 1]; j < binOffsets[bin + numRows]; ++j){
                    apply_force(parts[i], parts[binIndices[j]]);
                }
            }

            // right
            if (hasRight){
                for (int j = binOffsets[bin + numRows + 1]; j < binOffsets[bin + numRows + 2]; ++j){
                    apply_force(parts[i], parts[binIndices[j]]);
                }
            }
        }
    }

    // MOVE 
    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }

    // clear arrays
    for (int i = 0; i < totalBins; ++i){
        binCounts[i] = 0;
    }
}
