#include "RenderHelpers.h"

using namespace std;

void RenderHelpers::drawCircle(glm::vec2 center, float r) {
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(center.x, center.y);

    for (int i = 0; i <= 20; ++i) {
        float theta = 2.0f * PI * i / 20;
        glVertex2f(center.x + r * cos(theta), center.y + r * sin(theta));
    }

    glEnd();
}

void RenderHelpers::drawRadius(glm::vec2 center, float r) {
    glBegin(GL_LINE_LOOP);

    for (int i = 0; i <= 20; i++) {
        float theta = 2.0f * PI * i / 20;
        glVertex2f(center.x + r * cos(theta), center.y + r * sin(theta));
    }

    glEnd();
}


void RenderHelpers::drawArrow(glm::vec2 from, glm::vec2 to, float arrowHeadSize) {
    glBegin(GL_LINES);
    glVertex2f(from.x, from.y);
    glVertex2f(to.x, to.y);
    glEnd();

    glm::vec2 dir = glm::normalize(to - from);
    glm::vec2 perp = glm::vec2(-dir.y, dir.x);

    glm::vec2 tip = to;
    glm::vec2 left = tip - dir * arrowHeadSize + perp * arrowHeadSize * 0.5f;
    glm::vec2 right = tip - dir * arrowHeadSize - perp * arrowHeadSize * 0.5f;

    glBegin(GL_TRIANGLES);
    glVertex2f(tip.x, tip.y);
    glVertex2f(left.x, left.y);
    glVertex2f(right.x, right.y);
    glEnd();
}


/**
*    Allows us to render the final solution to the problem!
*
 */
void RenderHelpers::render(vector<vector<float>>& adjacencyList, vector<glm::vec2>& absolutePath, vector<glm::vec2>& nodePositionsReal, vector<int>& route) {
    glClear(GL_COLOR_BUFFER_BIT);

    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);

    // This will draw the absolute path taken by the drones to visit the sensors
    glColor3f(1.0f, 0.0f, 0.0f);
    for (int i = 1; i < route.size(); ++i) {
        glm::vec2 a = absolutePath[i-1];
        glm::vec2 b = absolutePath[i];
        drawArrow(a, b);
    }

    glColor3f(0.55f, 0.55f, 0.55f);
    glm::vec2 last = absolutePath[route[route.size() - 1]];
    glm::vec2 base = absolutePath[route[0]];
    drawArrow(last, base);
    glEnd();


    for (const auto& pos : nodePositionsReal) {
        glColor3f(0.2f, 0.6f, 1.0f);
        drawCircle(pos, 0.004f);

        glColor3f(0.2f, 1.0f, .2f);
        drawRadius(pos, SENSOR_RADIUS);
    }
}
