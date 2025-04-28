#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GLM/glm.hpp>
#include <iostream>
#include <vector>

struct Boid {
  float x, y;
  float vx, vy;
};

void updateBoids(float deltaTime, std::vector<Boid> &flock, float wAlign, float wCohesion, float wSeperate) {
    const float neighborRadius = 0.1f;
    const float separationDistance = 0.02f;
	const float baseSpeed = 0.3f;

    for (int i = 0; i < flock.size(); i++) {
        auto& boid = flock[i];

        // Average heading
        float avgHeadX = 0.0f;
        float avgHeadY = 0.0f;

        // Center of mass
        float cX = 0.0f;
        float cY = 0.0f;

        // Separation
        float sepX = 0.0f;
        float sepY = 0.0f;

        int numNeighbors = 0;

        for (int j = 0; j < flock.size(); j++) {
            if (i == j) continue;

            auto& neighbor = flock[j];

            float dx = neighbor.x - boid.x;
            float dy = neighbor.y - boid.y;
            float distSquared = dx * dx + dy * dy;

            if (distSquared < neighborRadius * neighborRadius) {
                numNeighbors++;

                // Average heading (alignment)
                avgHeadX += neighbor.vx;
                avgHeadY += neighbor.vy;

                // Center of mass (cohesion)
                cX += neighbor.x;
                cY += neighbor.y;

                // Separation
                if (distSquared < separationDistance * separationDistance) {
                    sepX -= (neighbor.x - boid.x);
                    sepY -= (neighbor.y - boid.y);
                }
            }
        }

        if (numNeighbors > 0) {
            avgHeadX /= numNeighbors;
            avgHeadY /= numNeighbors;
            cX /= numNeighbors;
            cY /= numNeighbors;

            // Alignment, Match neighbors' heading
            boid.vx += (avgHeadX - boid.vx) * wAlign * deltaTime;
            boid.vy += (avgHeadY - boid.vy) * wAlign * deltaTime;

            // Cohesion. Move toward center of mass
            boid.vx += (cX - boid.x) * wCohesion * deltaTime;
            boid.vy += (cY - boid.y) * wCohesion * deltaTime;

            // Separation, Move away from close neighbors
            boid.vx += sepX * wSeperate * deltaTime;
            boid.vy += sepY * wSeperate * deltaTime;
        }

		// REMOVE ME TO SEE THE "GRAVITY BALLS"
        float speed = std::sqrt(boid.vx * boid.vx + boid.vy * boid.vy);
        if (speed > 0.0001f) {
            boid.vx = (boid.vx / speed) * baseSpeed;
            boid.vy = (boid.vy / speed) * baseSpeed;
        }

        boid.vx += ((rand() / (float)RAND_MAX - 0.5f) * 0.005f);
		boid.vy += ((rand() / (float)RAND_MAX - 0.5f) * 0.005f);

        // Update position
        boid.x += boid.vx * deltaTime;
        boid.y += boid.vy * deltaTime;

        // Pac-Man wrap-around
        if (boid.x < -1.0f) boid.x = 1.0f;
        if (boid.x >  1.0f) boid.x = -1.0f;
        if (boid.y < -1.0f) boid.y = 1.0f;
        if (boid.y >  1.0f) boid.y = -1.0f;
    }
}


void drawBoids(std::vector<Boid>& flock) {
    glBegin(GL_TRIANGLES);

    for (const auto& b : flock) {
        // Find angle of velocity
        float angle = atan2(b.vy, b.vx); // Assuming Boid has vx, vy for velocity

        // Triangle shape in local space (pointing along +x)
        float size = 0.01f; // control overall triangle size
        float backSize = 0.005f; // how wide the base is

        glm::vec2 tip = glm::vec2( size,  0.0f);
        glm::vec2 left = glm::vec2(-size,  backSize);
        glm::vec2 right = glm::vec2(-size, -backSize);

        // Build rotation matrix manually
        float cosA = cos(angle);
        float sinA = sin(angle);

        auto rotate = [&](const glm::vec2& p) -> glm::vec2 {
            return glm::vec2(
                cosA * p.x - sinA * p.y,
                sinA * p.x + cosA * p.y
            );
        };

        // Rotate and translate points to face triangle right way.
        glm::vec2 p1 = rotate(tip)   + glm::vec2(b.x, b.y);
        glm::vec2 p2 = rotate(left)  + glm::vec2(b.x, b.y);
        glm::vec2 p3 = rotate(right) + glm::vec2(b.x, b.y);

        // Render
        glVertex2f(p1.x, p1.y);
        glVertex2f(p2.x, p2.y);
        glVertex2f(p3.x, p3.y);
    }

    glEnd();
}


int main(int argc, char** argv) {
  	int numBoids = 100;
	float wAlign = 0.5f;
    float wCohesion = 5.0f;
	float wSeperate = .60f;

	std::cout << "Usage: ./CudaOpenGLFlocking.exe <Num_Boids> <Alignment_Weight> <Cohesion_Weight> <Seperation_Weight> " << std::endl;
    if(argc > 1) numBoids = atoi(argv[1]);
	if(argc > 2) wAlign = atoi(argv[2]);
    if(argc > 3) wCohesion = atoi(argv[3]);
    if(argc > 4) wSeperate = atoi(argv[4]);

  	std::vector<Boid> flock;

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(1000, 1000, "CUDA OpenGL Flocking", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        return -1;
    }

        // Initialize boids
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    for (int i = 0; i < numBoids; ++i) {
        Boid b;
        b.x = (std::rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        b.y = (std::rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        b.vx = (((std::rand() / (float)RAND_MAX) - 0.5f) * 0.8f);
 		b.vy = (((std::rand() / (float)RAND_MAX) - 0.5f) * 0.8f);
        flock.push_back(b);
    }

    double lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
      	double currentTime = glfwGetTime();
    	float deltaTime = static_cast<float>(currentTime - lastTime);
    	lastTime = currentTime;

        glClear(GL_COLOR_BUFFER_BIT);
        updateBoids(deltaTime, flock, wAlign, wCohesion, wSeperate);
        drawBoids(flock);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}