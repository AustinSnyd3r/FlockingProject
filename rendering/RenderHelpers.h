#ifndef RENDERHELPERS_H
#define RENDERHELPERS_H
#include <GL\glew.h>
#include <vector>
#include <GLM/glm.hpp>
#include "../config.h"


class RenderHelpers {
public:
    static void drawCircle(glm::vec2 center, float r);
    static void drawRadius(glm::vec2 center, float r);
    static void drawArrow(glm::vec2 from, glm::vec2 to, float arrowHeadSize = 0.01f);
    static void render(std::vector<std::vector<float>>& adjacencyList, std::vector<glm::vec2>& absolutePath, std::vector<glm::vec2>& nodePositionsReal, std::vector<int>& route);
};



#endif //RENDERHELPERS_H
