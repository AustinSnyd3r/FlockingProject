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
#include <random>
#include <sstream>
#include <thread>
#include <chrono>
#include "config.h"
#include "./rendering/RenderHelpers.h"
#include <cuda.h>
#include <curand_kernel.h>
#include <vector>
#include <float.h>
#include <iostream>

using namespace std;

#define USE_CPU false


// NOTE: RIGHT NOW THIS IS THE MAX NUMBER OF CITIES THAT CAN BE USED. WE CAN INFLATE IT
// BUT WE REALLY SHOULD ADD IN DYNAMIC MEMORY ALLOCATION FOR IT INSTEAD IF POSSIBLE
// IT IS KIND OF A WASTE
#define MAX_N 1024
const float LEARNING_RATE = 0.01f;
const int MAX_ITER = 1000;
const float GRAD_EPSILON = 1e-4f;

vector<pair<float, float>> importDataset(string fileName);

vector<glm::vec2> computeNodePositions(int numNodes, const vector<pair<float, float>>& coords);
vector<vector<float>> convertToGraph(int dimension, const vector<pair<float, float>>& coords);
vector<glm::vec2> gradientDescentPathOptimization(const vector<int>& route, const vector<glm::vec2>& nodePositions);
bool isInsideCircle(float x, float y, float cx, float cy, float r);
float euclideanDistance(const glm::vec2& a, const glm::vec2& b);
float euclideanDistance(const pair<float, float>& p1, const pair<float, float>& p2);
float totalPathLength(const std::vector<glm::vec2>& path);
float distance(int from, int to, const vector<vector<float>>& adj);
float computePathLength(const vector<int>& path, const vector<vector<float>>& adj, int N);
vector<int> constructSolution(int baseStationIdx, const vector<vector<float>>& pheromones, const vector<vector<float>>& adj, mt19937& rng, int N);
std::vector<int> antColonyCUDA(const std::vector<std::vector<float>>& adj, int N, int baseStationIdx);

int main(int argc, char** argv) {

    //TODO: Take in command line args for this later.
    vector<pair<float, float>> coordinates = importDataset("C:\\Users\\asnyd\\CLionProjects\\CudaOpenGLFlocking\\datasets\\gridAltered.tsp");
    int N = coordinates.size();

    // Convert the coordinates into a adjacency list
    vector<vector<float>> graph = convertToGraph(N, coordinates);
    vector<glm::vec2> nodesVis = computeNodePositions(N, coordinates);

    vector<int> acoRoute = antColonyCUDA(graph, N, 0);

    vector<glm::vec2> finalRoutePos = gradientDescentPathOptimization(acoRoute, nodesVis);

    if (!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(900, 900, "DMRP Visualized", NULL, NULL);
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

    using Clock = std::chrono::high_resolution_clock;
    auto lastRender = Clock::now();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();  // Always responsive

        auto now = Clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastRender);

        if (elapsed.count() >= 1000) {
            RenderHelpers::render(graph, finalRoutePos, nodesVis, acoRoute);
            glfwSwapBuffers(window);
            lastRender = now;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));  // Light sleep to reduce CPU spin
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();


    return 0;
    //CUDA(argc, argv);
}

/**
 * Helper function to calculate the Euclidean distance between two points.
 * @param p1 The first x,y point
 * @param p2 The second x,y point
 * @return The distance between x,y.
 */
float euclideanDistance(const pair<float, float>& p1, const pair<float, float>& p2) {
    float dist = 0.0f;

    dist += pow(p1.first - p2.first, 2);
    dist += pow(p1.second - p2.second, 2);

    return sqrt(dist);
}

/**
 * Imports a traveling salesman type dataset, we use this with a manually input radius of communication
 * that is the same for all sensors. In the future, we should change to have each sensor have a different
 * radius. This would be more realistic.
 *
 * Sample datasets used in testing:
 * https://www.math.uwaterloo.ca/tsp/data/index.html
 *
 */
vector<pair<float, float>> importDataset(string fileName) {
    ifstream file(fileName);
    if (!file.is_open()) {
        cerr << "Failed to open file.\n";
        exit(1);
    }

    string line;
    vector<pair<float, float>> coordinates;

    bool readCoord = false;
    int dimension = 0;

    while(getline(file, line)) {
        // We have the dimension now.
        if(line.find("DIMENSION") != string::npos) {
            dimension = stoi(line.substr(line.find(":") + 1));
            coordinates.resize(dimension);
        }

        // We have reached the coordinate section of the dataset
        if(line.find("NODE_COORD_SECTION") != string::npos) {
            readCoord = true;
            continue;
        }

        // Add in the line that is like "id x y" into the coords.
        if(readCoord) {
            // Take the line, it will be split 3 times: id, x, y
            istringstream iss(line);
            int id;
            double x, y;
            if(!(iss >> id >> x >> y)) break;
            coordinates[id - 1] = make_pair(x, y);
        }
    }

    return coordinates;
}

vector<vector<float>> convertToGraph(const int dimension, const vector<pair<float, float>>& coords) {
    // Now we want to actually make the graph representation now, in adjacency list.
    vector<vector<float>> dist(dimension, vector<float>(dimension));
    for(int i = 0; i < dimension; i++) {
        for(int j = 0; j < dimension; j++) {
            dist[i][j] = euclideanDistance(coords[i], coords[j]);
        }
    }

    return dist;
}


/**
 * Convert the actual longitude and latitudes into a version we can render.
 * @param numNodes
 * @param coordinates
 * @return
 */
vector<glm::vec2> computeNodePositions(const int numNodes, const vector<pair<float, float>>& coordinates) {
    vector<glm::vec2> nodePositions(0);
    float minX = numeric_limits<float>::infinity();
    float maxX = -numeric_limits<float>::infinity();

    float minY = numeric_limits<float>::infinity();
    float maxY = -numeric_limits<float>::infinity();

    // Get mins and maxes of the "space", I want to plot how it actually looks.
    for(int i = 0; i < numNodes; i++) {
        maxX = max(maxX, coordinates[i].first);
        minX = min(minX, coordinates[i].first);

        maxY = max(maxY, coordinates[i].second);
        minY = min(minY, coordinates[i].second);
    }

    // Some buffer on the border
    maxX += 50;
    maxY += 50;
    minX -= 50;
    minY -= 50;

    // Normalize the coordinates to a 0-1 range to allow visualization
    for (int i = 0; i < numNodes; ++i) {
        float x = (coordinates[i].first - minX) / (maxX - minX);
        float y = (coordinates[i].second - minY) / (maxY - minY);
        nodePositions.emplace_back(x, y);
    }

    return nodePositions;
}



bool isInsideCircle(const float x, const float y, const float cx, const float cy, const float r) {
    const float dx = x - cx;
    const float dy = y - cy;
    return (dx * dx + dy * dy) <= (r * r);
}

float euclideanDistance(const glm::vec2& a, const glm::vec2& b) {
    return glm::length(b - a);
}


/**
 * Simple helper function that computes the distance that the path takes overall.
 * @param path The vector of glm::vec2, these are (x,y) coordinates.
 * @return The total length of the path in some arbitary unit.
 */
float totalPathLength(const std::vector<glm::vec2>& path) {
    float total = 0.0f;

    for (int i = 0; i + 1 < path.size(); i++) {
        total += euclideanDistance(path[i], path[i + 1]);
    }

    return total;
}


std::vector<glm::vec2> gradientDescentPathOptimization(const std::vector<int>& route, const std::vector<glm::vec2>& nodePositions) {

    if (route.empty()) return {};

    std::vector<glm::vec2> path(route.size());
    for (size_t i = 0; i < route.size(); ++i) {
        path[i] = nodePositions[route[i]];
    }

    cout << "Route has same length as path: " << (path.size() == route.size()) << endl;
    for(int i = 1; i < route.size(); ++i) {
        auto p = path[i];
        cout << route[i] << " Location: ";
        cout << p.x << ", " << p.y << endl;
    }

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        float prevCost = totalPathLength(path);

        for (int i = 1; i < static_cast<int>(route.size()) - 1; ++i) {
            glm::vec2 prev = path[i - 1];
            glm::vec2 next = path[i + 1];
            glm::vec2 center = nodePositions[route[i]];

            glm::vec2 curr = path[i];
            glm::vec2 grad = glm::normalize(curr - prev) + glm::normalize(curr - next);
            glm::vec2 proposed = curr - LEARNING_RATE * grad;

            // Clamp proposed point to lie within the radius from center in 2D
            glm::vec2 offset = proposed - center;
            float dist = glm::length(offset);
            if (dist > SENSOR_RADIUS) {
                offset = (offset / dist) * SENSOR_RADIUS; // normalize and scale
                proposed = center + offset;
            }

            // Only accept if it improves local segment cost
            float oldCost = glm::length(curr - prev) + glm::length(curr - next);
            float newCost = glm::length(proposed - prev) + glm::length(proposed - next);

            if (newCost < oldCost) {
                path[i] = proposed;
            }
        }

        float newCost = totalPathLength(path);
        if (std::abs(newCost - prevCost) < GRAD_EPSILON) break;
    }

    // Final enforcement of radius constraint (double safety)
    for (int i = 0; i < static_cast<int>(route.size()); ++i) {
        glm::vec2 center = nodePositions[route[i]];
        glm::vec2 offset = path[i] - center;
        float dist = glm::length(offset);
        if (dist > SENSOR_RADIUS) {
            offset = (offset / dist) * SENSOR_RADIUS;
            path[i] = center + offset;
        }
    }

    return path;
}


/**
 *
 * I SHOULD SEPERATE THE FUNCTIONALITY FOR THE ANT COLONY RELATED STUFF INTO A SEPERATE FILE, BUT NOT YET!
 *
 */
#define ANTS 100   // number of ants in each generation
#define ITERATIONS 100 // num of iterations
#define ALPHA 1.0f  // pheromone importance
#define BETA 5.0f   // distance importance
#define EVAPORATION 0.5f  // rate of pheremone evaporation
#define Q 100.0f  // weight

/**
 * This does same thing as if we had "vector<vector<float>>& adj"
 *  and we returned adj[from][to].
 *
 *  Since cuda is flattened arrays, we use this indexing scheme.
 */
__device__ float distance(int from, int to, float* adj, int N) {
    return adj[from * N + to];
}

/**
 * Used to be:
 * float computePathLength(const vector<int>& path, const vector<vector<float>>& adj, int N)
 *
 */
__device__ float computePathLength(int* path, float* adj, int N) {
    float len = 0.0f;
    for (int i = 0; i < N; i++) {
        int from = path[i];
        int to = path[(i + 1) % N];
        len += distance(from, to, adj, N);
    }
    return len;
}


__global__ void initRNG(curandState* states, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < ANTS) {
        curand_init(seed, id, 0, &states[id]);
    }
}


__global__ void constructSolutions(
    int N, int baseStationIdx, float* pheromones, float* adj,
    int* paths, float* lengths, float* probabilities, curandState* rngStates){

    int antIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (antIdx >= ANTS) return;

    float* myProbabilities = &probabilities[antIdx * N];

    int* path = &paths[antIdx * N];
    bool visited[MAX_N] = {0};

    path[0] = baseStationIdx;
    visited[baseStationIdx] = true;

    curandState localState = rngStates[antIdx];

    for (int step = 1; step < N; ++step) {
        int current = path[step - 1];
        float sum = 0.0f;

        for (int j = 0; j < N; ++j) {
            if (!visited[j]) {
                float tau = powf(pheromones[current * N + j], ALPHA);
                float eta = powf(1.0f / distance(current, j, adj, N), BETA);
                myProbabilities[j] = tau * eta;
                sum += probabilities[j];
            } else {
                myProbabilities[j] = 0.0f;
            }
        }

        float pick = curand_uniform(&localState) * sum;
        float cumulative = 0.0f;
        int next = -1;

        for (int j = 0; j < N; ++j) {
            cumulative += myProbabilities[j];
            if (pick <= cumulative && !visited[j]) {
                next = j;
                break;
            }
        }

        if (next == -1) {
            for (int j = 0; j < N; ++j) {
                if (!visited[j]) {
                    next = j;
                    break;
                }
            }
        }

        path[step] = next;
        visited[next] = true;
    }

    lengths[antIdx] = computePathLength(path, adj, N);
    rngStates[antIdx] = localState;
}

std::vector<int> antColonyCUDA(const std::vector<std::vector<float>>& adj, int N, int baseStationIdx) {
    float* d_adj, *d_pheromones;
    int* d_paths;
    float* d_lengths;
    curandState* d_rngStates;

    int size = N * N * sizeof(float);
    cudaMalloc(&d_adj, size);
    cudaMalloc(&d_pheromones, size);
    cudaMalloc(&d_paths, sizeof(int) * N * ANTS);
    cudaMalloc(&d_lengths, sizeof(float) * ANTS);
    cudaMalloc(&d_rngStates, sizeof(curandState) * ANTS);

    float* adj_flat = new float[N * N];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            adj_flat[i * N + j] = adj[i][j];

    cudaMemcpy(d_adj, adj_flat, size, cudaMemcpyHostToDevice);

    std::vector<float> pheromone(N * N, 1.0f);
    cudaMemcpy(d_pheromones, pheromone.data(), size, cudaMemcpyHostToDevice);

    initRNG<<<(ANTS + 31) / 32, 32>>>(d_rngStates, time(NULL));

    std::vector<int> bestPath(N);
    float bestLength = FLT_MAX;

    std::vector<float> lengths(ANTS);
    std::vector<int> paths(ANTS * N);

    float* d_probabilities;
    cudaMalloc(&d_probabilities, sizeof(float) * N * ANTS);

    for (int iter = 0; iter < ITERATIONS; ++iter) {

        constructSolutions<<<(ANTS + 31)/32, 32>>>(N, baseStationIdx, d_pheromones, d_adj, d_paths, d_lengths, d_probabilities, d_rngStates);

        cudaMemcpy(lengths.data(), d_lengths, sizeof(float) * ANTS, cudaMemcpyDeviceToHost);
        cudaMemcpy(paths.data(), d_paths, sizeof(int) * N * ANTS, cudaMemcpyDeviceToHost);

        for (int i = 0; i < N * N; ++i)
            pheromone[i] *= (1.0f - EVAPORATION);

        for (int k = 0; k < ANTS; ++k) {
            float len = lengths[k];
            if (len < bestLength) {
                bestLength = len;
                std::copy(paths.begin() + k * N, paths.begin() + (k + 1) * N, bestPath.begin());
            }

            for (int i = 0; i < N; ++i) {
                int from = paths[k * N + i];
                int to = paths[k * N + (i + 1) % N];
                pheromone[from * N + to] += Q / len;
                pheromone[to * N + from] += Q / len;
            }
        }

        cudaMemcpy(d_pheromones, pheromone.data(), size, cudaMemcpyHostToDevice);
    }

    bestPath.push_back(baseStationIdx);

    cudaFree(d_adj);
    cudaFree(d_pheromones);
    cudaFree(d_paths);
    cudaFree(d_lengths);
    cudaFree(d_probabilities);
    cudaFree(d_rngStates);
    delete[] adj_flat;

    return bestPath;
}