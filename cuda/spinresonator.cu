
#include <stdint.h>
#include "float3.h"
#include "amul.h"

// Add resonator field to B.
extern "C" __global__ void
addresonatorfield(float* __restrict__  Bx, float* __restrict__  By, float* __restrict__  Bz,
                       float voltage,
                       float current, 
                       float brf,
                       int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        Bx[i] += brf * current;
    }
}

