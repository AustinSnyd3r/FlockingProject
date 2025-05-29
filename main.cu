#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GLM/glm.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <vector>
#include <limits>
#include <cstdlib>
#include <utility>
#include <ctime>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
using namespace std;

#define USE_CPU false
#define PI 3.14159265358979323846f

cudaGraphicsResource* cudaBoidVBOResource;
GLuint boidVBO;

__host__ void createBoidVBO(GLuint* vbo, int numBoids);




vector<pair<float, float>> importDataset(std::string fileName);
void render(vector<vector<float>> adjacencyList, vector<glm::vec2> nodePositions);
void drawCircle(glm::vec2 center, float r);
vector<glm::vec2> computeNodePositions(int numNodes, vector<pair<float, float>> coords);
vector<vector<float>> convertToGraph(int dimension, vector<pair<float, float>> coords);


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

/**s
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
    if (argc > 1) numBoids = std::atoi(argv[1]);
    if (argc > 2) wAlign = std::atof(argv[2]);
    if (argc > 3) wCohesion = std::atof(argv[3]);
    if (argc > 4) wSeperate = std::atof(argv[4]);

    // The grid for how we will split the screen to hash locality of boids.
    // For example, we don't really care about two far away boids interacting, since it will be minimal effect.
    int gridWidth = 8, gridHeight = 8;
    int totalBins = gridWidth * gridHeight;
    std::vector<float4> cpuBoids(numBoids);
    std::srand((unsigned)std::time(0));
    for (int i = 0; i < numBoids; ++i) {
        cpuBoids[i] = make_float4(
            ((std::rand() / (float)RAND_MAX) * 2.0f - 1.0f),
            ((std::rand() / (float)RAND_MAX) * 2.0f - 1.0f),
            ((std::rand() / (float)RAND_MAX) - 0.5f) * 0.2f,
            ((std::rand() / (float)RAND_MAX) - 0.5f) * 0.2f);
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
        double currentTime = glfwGetTime();
        float deltaTime = static_cast<float>(currentTime - lastTime);
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



int main(int argc, char** argv) {
    // Removed CPU version entirely

    vector<pair<float, float>> coordinates = importDataset("C:\\Users\\asnyd\\CLionProjects\\CudaOpenGLFlocking\\datasets\\dj38.tsp");
    int N = coordinates.size();
    vector<vector<float>> graph = convertToGraph(N, coordinates);
    vector<glm::vec2> nodesVis = computeNodePositions(N, coordinates);

    if (!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(1200, 1200, "Graph Visualizer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glewInit();

    // Setup 2D orthographic projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    while (!glfwWindowShouldClose(window)) {
        render(graph, nodesVis);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();


    return 0;
    //CUDA(argc, argv);
}










// Data Mule Routing Problem (with Limited Autonomy)

/**
 * Essentially the papers read aim to travel to wireless sensors with drones.
 * These sensors have a set radius that they can communicate with the drone to transmit information.
 * We need to leave from the "base station" and travel in a path that collects all the sensor information
 * in the smallest amount of time.
 *
 *
 * In terms of time, both training time to find the solution, and the optimality of our paths matter heavily.
 * In wild-land rescue, a model that trains for 10 hours could cause someone to never be found or even die.
 *
 *
 * Since we have sensors with a radius of communication, there are possible overlaps between sensor
 *
 *
 * I cannot find the online datasets that they used because they are locked inside a springer publication
 *
 * Instead, i have some TSP datasets that I will use and add in the radius manually. Additionally, the first
 * node in the set will be the "base"
 */
void createDMRP(int numSensors, float radius, int numDrones) {




}


// Remember to ask Dr. Davendra:
// Ant Colony Optimization with TSP / TSP with neighborhoods
// TSP common approaches in general.
// Papers about common algorithms in Scientific Computing and their pros and cons.


float euclideanDistance(const std::pair<float, float>& p1, const std::pair<float, float>& p2) {
    float dist = 0.0f;

    dist += pow(p1.first - p2.first, 2);
    dist += pow(p1.second - p2.second, 2);

    return sqrt(dist);
}

/**
 * Imports a traveling salesman type dataset.
 */
vector<pair<float, float>> importDataset(std::string fileName) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Failed to open file.\n";
        exit(1);
    }

    std::string line;
    std::vector<std::pair<float, float>> coordinates;

    bool readCoord = false;
    int dimension = 0;

    while(std::getline(file, line)) {
        // We have the dimension now.
        if(line.find("DIMENSION") != std::string::npos) {
            dimension = std::stoi(line.substr(line.find(":") + 1));
            coordinates.resize(dimension);
        }

        // We have reached the coordinate section of the dataset
        if(line.find("NODE_COORD_SECTION") != std::string::npos) {
            readCoord = true;
            continue;
        }

        // Add in the line that is like id x y into the coords.
        if(readCoord) {
            // Take the line, it will be split 3 times: id, x, y
            std::istringstream iss(line);
            int id;
            double x, y;
            if(!(iss >> id >> x >> y)) break;
            coordinates[id - 1] = std::make_pair(x, y);
        }
    }

    return coordinates;
}

vector<vector<float>> convertToGraph(int dimension, vector<pair<float, float>> coordinates) {
    // Now we want to actually make the graph representation now, in adjacency list.
    std::vector<std::vector<float>> dist(dimension, std::vector<float>(dimension));
    for(int i = 0; i < dimension; i++) {
        for(int j = 0; j < dimension; j++) {
            dist[i][j] = euclideanDistance(coordinates[i], coordinates[j]);
        }
    }

    // Lets see if it was made correctly.
    for(const auto& coord : dist) {
        for(int i = 0; i < dimension; i++) {
            std::cout << coord[i] << ", ";
        }
        std::cout << std::endl;
    }

    return dist;
}


/**
 * Convert the actual longitude and latitudes into a version we can render.
 * @param N
 * @param coordinates
 * @return
 */
vector<glm::vec2> computeNodePositions(int N, vector<pair<float, float>> coordinates) {
    vector<glm::vec2> nodePositions(0);
    float minX = numeric_limits<float>::infinity();
    float maxX = -numeric_limits<float>::infinity();

    float minY = numeric_limits<float>::infinity();
    float maxY = -numeric_limits<float>::infinity();

    // Get mins and maxes of the "space", i want to plot how it actually looks.
    for(int i = 0; i < N; i++) {
        maxX = std::max(maxX, coordinates[i].first);
        minX = std::min(minX, coordinates[i].first);

        maxY = std::max(maxY, coordinates[i].second);
        minY = std::min(minY, coordinates[i].second);
    }

    // Some buffer on the border
    maxX += 50;
    maxY += 50;
    minX -= 50;
    minY -= 50;

    // Normalize the coordinates to a 0-1 range to allow visualization
    for (int i = 0; i < N; ++i) {
        float x = (coordinates[i].first - minX) / (maxX - minX);
        float y = (coordinates[i].second - minY) / (maxY - minY);
        nodePositions.emplace_back(x, y);
    }

    return nodePositions;
}

void drawCircle(glm::vec2 center, float r) {
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(center.x, center.y);

    for (int i = 0; i <= 20; ++i) {
        float theta = 2.0f * PI * i / 20;
        glVertex2f(center.x + r * cos(theta), center.y + r * sin(theta));
    }

    glEnd();
}

void render(vector<vector<float>> adjacencyList, vector<glm::vec2> nodePositions) {
    glClear(GL_COLOR_BUFFER_BIT);

    glColor3f(1.0f, 1.0f, 1.0f); // white lines
    glBegin(GL_LINES);
    for (int i = 0; i < adjacencyList.size(); ++i) {
        for (int j = 0; j < adjacencyList[i].size(); j++) {

            glm::vec2 a = nodePositions[i];
            glm::vec2 b = nodePositions[j];
            glVertex2f(a.x, a.y);
            glVertex2f(b.x, b.y);

        }
    }
    glEnd();

    glColor3f(0.2f, 0.6f, 1.0f); // blue nodes
    for (const auto& pos : nodePositions) {
        drawCircle(pos, 0.01f);
    }
}