#include <cuda_runtime.h>

/**
  The struct for a single boid.

  x, y, z position

  x, y, z velocity.


struct Boid {
  float x, y, z;
  float velX, velY, velZ;
};

*/

__global__ void updateBoids(float deltaTime, float4* boids, float wAlign, float wCohesion, float wSeperate){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const float seperationDistance = 0.02f;
  const float baseSpeed = 0.3f;

  // Get the boid that this thread represents.
  float4 boid = boids[tid];



}