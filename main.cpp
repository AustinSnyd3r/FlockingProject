#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GLM/glm.hpp>
#include <iostream>
#include <vector>
#include <utility>

struct Boid {
  float x, y;
  float vx, vy;
};

struct Point {
  float x, y;
  Point(float x_, float y_) : x(x_), y(y_){}
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

// Cpu based version, need to adapt for use inside of cuda.
int getBinIndex(const &pos, int gridWidth, int gridHeight) {
    float clampedX = std::max(-1.0f, std::min(1.0f, pos.x));
    float clampedY = std::max(-1.0f, std::min(1.0f, pos.y));

    // Convert opengl based position to 0, 1 range
    float normX = (clampedX + 1.0f) / 2.0f;
    float normY = (clampedY + 1.0f) / 2.0f;

    // scale to grid coordinates
    int binX = std::min(static_cast<int>(normX * gridWidth), gridWidth - 1);
    int binY = std::min(static_cast<int>(normY * gridHeight), gridHeight - 1);

    // flatten 2D bin index to 1-d
    return binY * gridWidth + binX;
}

// Imagine it is called with -1, 1 as the bottomleft and topright
void divideToBins(char numBins, std::vector<std::pair<Point, Point>> &binBounds, Point bottomLeft, Point topRight) {
  if(numBins == 1){
    binBounds.push_back(std::pair<Point, Point>(bottomLeft, topRight));
    return;
  }

  if(numBins < 1){
     std::cerr << "Number of bins must be an even number. Function recieved: "<< numBins << std::endl;
     exit(-1);
  }
  // We know the edges of the screen are -1, -1 (bottom left), 1, -1 (bottom right) -1, 1 (top left) 1, 1 (top right)
  // If we want 4 bins, then center of all of these bins will be 0, 0
  // Essentially we divided the screen into 2 on horizontal and vertical

  // The center point.
  Point center((bottomLeft.x + topRight.x) / 2.0f, (bottomLeft.y + topRight.y) / 2.0f);

  divideToBins(numBins / 4, binBounds, bottomLeft, center);
  divideToBins(numBins / 4, binBounds, Point(center.x, bottomLeft.y), Point(topRight.x, center.y));
  divideToBins(numBins / 4, binBounds, Point(bottomLeft.x, center.y), Point(center.x, topRight.y));
  divideToBins(numBins / 4, binBounds, center, topRight);

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

void drawBins(std::vector<std::pair<Point, Point>> &bins) {
	for(const std::pair<Point, Point>& b : bins) {
        glBegin(GL_LINE_LOOP);
    	glVertex2f(b.first.x, b.first.y); // Bottom Left
		glVertex2f(b.second.x, b.first.y); // Bottom Right
		glVertex2f(b.second.x, b.second.y); // Top Right
		glVertex2f(b.first.x, b.second.y); // Top Left


int CPU(int argc, char** argv) {
  	std::vector<std::pair<Point, Point>> binBounds;
    Point bottomLeft(-1, -1);
 	Point topRight(1, 1);

  	divideToBins(64, binBounds, bottomLeft, topRight);
	for(const auto& b : binBounds) {
    	std::cout << "Bottom Left " << std::endl;
    	std::cout << b.first.x << ", " << b.first.y << std::endl;

		std::cout << "Top Right " << std::endl;
    	std::cout << b.second.x << ", " << b.second.y << std::endl;
	}

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
        drawBins(binBounds);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}