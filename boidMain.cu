//
// Created by asnyd on 6/11/2025.
//
cudaGraphicsResource* cudaBoidVBOResource;
GLuint boidVBO;

__host__ void createBoidVBO(GLuint* vbo, int numBoids);


int CUDA(int argc, char** argv);

/**
 *
 * @param x
 * @param y
 * @param gridWidth
 * @param gridHeight
 * @return
 */
__device__ int getBinIndex(float x, float y, int gridWidth, int gridHeight) {
    float normX = (x + 1.0f) * 0.5f;
    float normY = (y + 1.0f) * 0.5f;

    int ix = min(max(int(normX * gridWidth), 0), gridWidth - 1);
    int iy = min(max(int(normY * gridHeight), 0), gridHeight - 1);

    // Return the index of the bin the boid is in.
    return iy * gridWidth + ix;
}

/**
 *
 * @param boids
 * @param boidBinIndices
 * @param numBoids
 * @param gridWidth
 * @param gridHeight
 */
__global__ void assignBoidsToBins(float4* boids, int* boidBinIndices, int numBoids, int gridWidth, int gridHeight) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numBoids) return;
    float4 b = boids[tid];
    int binIdx = getBinIndex(b.x, b.y, gridWidth, gridHeight);
    boidBinIndices[tid] = binIdx;
}


/**
 *
 * @param boidBinIndices
 * @param binStart
 * @param binEnd
 * @param numBoids
 */
__global__ void computeBinRanges(int* boidBinIndices, int* binStart, int* binEnd, int numBoids) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numBoids) return;

    int currentBin = boidBinIndices[tid];

    if (tid == 0 || boidBinIndices[tid - 1] != currentBin)
        binStart[currentBin] = tid;

    if (tid == numBoids - 1 || boidBinIndices[tid + 1] != currentBin)
        binEnd[currentBin] = tid + 1;
}


/**
 *
 * @param deltaTime
 * @param boids
 * @param wAlign
 * @param wCohesion
 * @param wSeperate
 * @param numBoids
 */
__global__ void updateBoids(float deltaTime, float4* boids, float wAlign, float wCohesion, float wSeperate, int numBoids) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numBoids) return;

    // Variable setup.
    const float neighborRadius = 0.1f;
    const float separationDistance = 0.02f;
    const float baseSpeed = 0.3f;

    // Get this threads boid
    float4 self = boids[tid];

    float avgHeadX = 0.0f, avgHeadY =0.0f;
    float centerX = 0.0f, centerY = 0.0f;
    float sepX = 0.0f, sepY = 0.0f;

    int neighbors = 0;

    // Go through and calculate the effect of ofther boids to this one
    for (int j = 0; j < numBoids; ++j) {
        if (j == tid) continue;
        float4 other = boids[j];

        // Distance to other boids
        float dx = other.x - self.x;
        float dy = other.y - self.y;
        float distSq = dx * dx + dy * dy;

        // only calculate the effect if it is within the sight of this boid
        if (distSq < neighborRadius * neighborRadius) {
            neighbors++;

            // turn boid toward header and mass of other boid
            avgHeadX += other.z;
            avgHeadY += other.w;
            centerX += other.x;
            centerY += other.y;

            // Apply seperation
            if (distSq < separationDistance * separationDistance) {
                sepX -= dx;
                sepY -= dy;
            }
        }
    }

    if (neighbors > 0) {
        avgHeadX /= neighbors;
        avgHeadY /= neighbors;
        centerX /= neighbors;
        centerY /= neighbors;
        self.z += (avgHeadX - self.z) * wAlign * deltaTime;
        self.w += (avgHeadY - self.w) * wAlign * deltaTime;
        self.z += (centerX - self.x) * wCohesion * deltaTime;
        self.w += (centerY - self.y) * wCohesion * deltaTime;
        self.z += sepX * wSeperate * deltaTime;
        self.w += sepY * wSeperate * deltaTime;
    }

    float speed = sqrtf(self.z * self.z + self.w * self.w);
    if (speed > 0.0001f) {
        self.z = (self.z / speed) * baseSpeed;
        self.w = (self.w / speed) * baseSpeed;
    }

    // Update based on dt
    self.x += self.z * deltaTime;
    self.y += self.w * deltaTime;

    // do the pacman effect when boid flies to edgfe of screen.
    if (self.x < -1.0f) self.x = 1.0f;
    if (self.x > 1.0f) self.x = -1.0f;
    if (self.y < -1.0f) self.y = 1.0f;
    if (self.y > 1.0f) self.y = -1.0f;

    // Update the boid in the mem
    boids[tid] = self;
}

/**
 * Creates vertex object buffer, allows us to render our boids in between time steps with openGL
 * @param vbo   The buffer
 * @param numBoids  Number of boids needed
 */
__host__ void createBoidVBO(GLuint* vbo, int numBoids) {
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * numBoids, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&cudaBoidVBOResource, *vbo, cudaGraphicsMapFlagsNone);
}


/**
 * THIS IS THE MAIN ACTION!!! Includes the loop that is calling our CUDA functions and updating the screen by using VBO shared
 * between openGL and CUDA
 * @param argc
 * @param argv
 * @return
 */
int CUDA(int argc, char** argv) {
    if (!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(800, 800, "CUDA OpenGL Boids", nullptr, nullptr);
    if (!window) return -1;

    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK) return -1;

    int numBoids = 1024;
    float wAlign = 0.5f, wCohesion = 5.0f, wSeperate = 0.6f;

    // Take command line input for boids and weights.
    if (argc > 1) numBoids = atoi(argv[1]);
    if (argc > 2) wAlign = atof(argv[2]);
    if (argc > 3) wCohesion = atof(argv[3]);
    if (argc > 4) wSeperate = atof(argv[4]);

    // The grid for how we will split the screen to hash locality of boids.
    // For example, we don't really care about two far away boids interacting, since it will be minimal effect.
    int gridWidth = 8, gridHeight = 8;
    int totalBins = gridWidth * gridHeight;
    vector<float4> cpuBoids(numBoids);
    srand((unsigned)time(0));
    for (int i = 0; i < numBoids; ++i) {
        cpuBoids[i] = make_float4(
            ((rand() / (float)RAND_MAX) * 2.0f - 1.0f),
            ((rand() / (float)RAND_MAX) * 2.0f - 1.0f),
            ((rand() / (float)RAND_MAX) - 0.5f) * 0.2f,
            ((rand() / (float)RAND_MAX) - 0.5f) * 0.2f);
    }

    // Device pointers
    float4* devBoids;
    int* devBoidBinIdx;
    int* devBinStart;
    int* devBinEnd;

    // VBO for rendering with the same mem as cuda uses
    createBoidVBO(&boidVBO, numBoids);

    // Allocate memory for gpu
    cudaMalloc(&devBoids, sizeof(float4) * numBoids);
    cudaMalloc(&devBoidBinIdx, sizeof(int) * numBoids);
    cudaMalloc(&devBinStart, sizeof(int) * totalBins);
    cudaMalloc(&devBinEnd, sizeof(int) * totalBins);

    // Copy the initial boids ove to the gpu
    cudaMemcpy(devBoids, cpuBoids.data(), sizeof(float4) * numBoids, cudaMemcpyHostToDevice);

    double lastTime = glfwGetTime();

    // MAIN LOOP, EACH ITERATION IS A TIME STEP!!!!
    while (!glfwWindowShouldClose(window)) {
        const double currentTime = glfwGetTime();
        const auto deltaTime = static_cast<float>(currentTime - lastTime);
        lastTime = currentTime;

        float4* devBoidPos;
        size_t size;
        cudaGraphicsMapResources(1, &cudaBoidVBOResource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&devBoidPos, &size, cudaBoidVBOResource);

        int blockSize = 256;
        int gridSize = (numBoids + blockSize - 1) / blockSize;

        // Split boids into the local bins on the screen
        assignBoidsToBins<<<gridSize, blockSize>>>(devBoids, devBoidBinIdx, numBoids, gridWidth, gridHeight);
        thrust::device_ptr<int> binIdxPtr(devBoidBinIdx);
        thrust::device_ptr<float4> boidPtr(devBoids);
        thrust::sort_by_key(binIdxPtr, binIdxPtr + numBoids, boidPtr);

        // Compute the range of the bins so we know
        computeBinRanges<<<gridSize, blockSize>>>(devBoidBinIdx, devBinStart, devBinEnd, numBoids);
        updateBoids<<<gridSize, blockSize>>>(deltaTime, devBoids, wAlign, wCohesion, wSeperate, numBoids);

        cudaMemcpy(devBoidPos, devBoids, sizeof(float4) * numBoids, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cudaBoidVBOResource, 0);

        // OpenGL clear and draw updated boids
        glClear(GL_COLOR_BUFFER_BIT);
        glBindBuffer(GL_ARRAY_BUFFER, boidVBO);
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(2, GL_FLOAT, sizeof(float4), 0);
        glDrawArrays(GL_POINTS, 0, numBoids);
        glDisableClientState(GL_VERTEX_ARRAY);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaGraphicsUnregisterResource(cudaBoidVBOResource);
    cudaFree(devBoids);
    cudaFree(devBoidBinIdx);
    cudaFree(devBinStart);
    cudaFree(devBinEnd);
    glDeleteBuffers(1, &boidVBO);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}