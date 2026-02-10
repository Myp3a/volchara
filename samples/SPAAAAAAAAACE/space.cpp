#include <chrono>
#include <random>
#include <vector>

#include <renderer.hpp>

volchara::Renderer renderer{};
std::mt19937 rnd(std::chrono::time_point<std::chrono::system_clock>().time_since_epoch().count());
std::uniform_real_distribution<> dis(0.2f, 1.0f);


class Planet : public volchara::Object {
    public:
    float speed;
    Planet(volchara::Renderer& renderer, float scale, float speed, float distance, std::filesystem::path texturePath) : volchara::Object(volchara::GLTFModel::fromFile(renderer, renderer.getResourceDir() / "models/sphere.glb")) {
        replaceTextures(texturePath);
        this->transform.scaling = {scale, scale, scale};
        this->children[1]->transform.position.right(distance/scale);
        this->speed = speed;
    }
};

void rotateAround(volchara::Object* obj, float passedSeconds, std::set<int> pressedKeys) {
    Planet* p = static_cast<Planet*>(obj);
    p->transform.rotation.left(passedSeconds * p->speed);
}

void bindCamera(std::vector<Planet*> planets, int* currentBound, volchara::Object* obj, float passedSeconds, std::set<int> pressedKeys) {
    if (pressedKeys.contains(GLFW_KEY_W) || pressedKeys.contains(GLFW_KEY_A) || pressedKeys.contains(GLFW_KEY_S) || pressedKeys.contains(GLFW_KEY_D) || pressedKeys.contains(GLFW_KEY_Q) || pressedKeys.contains(GLFW_KEY_E)) {
        *currentBound = -1;
    }
    if (pressedKeys.contains(GLFW_KEY_1)) {
        *currentBound = 0;
    }
    if (pressedKeys.contains(GLFW_KEY_2)) {
        *currentBound = 1;
    }
    if (pressedKeys.contains(GLFW_KEY_3)) {
        *currentBound = 2;
    }
    if (pressedKeys.contains(GLFW_KEY_4)) {
        *currentBound = 3;
    }
    if (pressedKeys.contains(GLFW_KEY_5)) {
        *currentBound = 4;
    }
    if (pressedKeys.contains(GLFW_KEY_6)) {
        *currentBound = 5;
    }
    if (pressedKeys.contains(GLFW_KEY_7)) {
        *currentBound = 6;
    }
    if (pressedKeys.contains(GLFW_KEY_8)) {
        *currentBound = 7;
    }
    if (pressedKeys.contains(GLFW_KEY_9)) {
        *currentBound = 8;
    }
    if (*currentBound > -1) {
        obj->renderer->camera.transform.translation = planets[*currentBound]->children[1]->transform.modelMatrix()[3];
        obj->renderer->camera.transform.position.backward(2);
        obj->renderer->camera.transform.position.up(2);
    }
}

int main() {
    Planet sun = Planet(renderer, 1, 0, 0, renderer.getResourceDir() / "textures/yellow.png");
    Planet mercury = Planet(renderer, 0.25, dis(rnd), 2, renderer.getResourceDir() / "textures/violet.png");
    mercury.frameCallbacks.push_back(rotateAround);
    Planet venus = Planet(renderer, 0.5, dis(rnd), 3, renderer.getResourceDir() / "textures/orange.png");
    venus.frameCallbacks.push_back(rotateAround);
    Planet earth = Planet(renderer, 0.40, dis(rnd), 5, renderer.getResourceDir() / "textures/green.png");
    earth.frameCallbacks.push_back(rotateAround);
    Planet mars = Planet(renderer, 0.35, dis(rnd), 6, renderer.getResourceDir() / "textures/red.png");
    mars.frameCallbacks.push_back(rotateAround);
    Planet jupiter = Planet(renderer, 0.75, dis(rnd), 8, renderer.getResourceDir() / "textures/purple.png");
    jupiter.frameCallbacks.push_back(rotateAround);
    Planet saturn = Planet(renderer, 0.5, dis(rnd), 10, renderer.getResourceDir() / "textures/yellow.png");
    saturn.frameCallbacks.push_back(rotateAround);
    Planet uranus = Planet(renderer, 0.45, dis(rnd), 11, renderer.getResourceDir() / "textures/cyan.png");
    uranus.frameCallbacks.push_back(rotateAround);
    Planet neptune = Planet(renderer, 0.40, dis(rnd), 12, renderer.getResourceDir() / "textures/lightblue.png");
    neptune.frameCallbacks.push_back(rotateAround);
    Planet pluto = Planet(renderer, 0.15, dis(rnd), 13, renderer.getResourceDir() / "textures/blue.png");
    pluto.frameCallbacks.push_back(rotateAround);
    renderer.addObject(&sun);
    renderer.addObject(&mercury);
    renderer.addObject(&venus);
    renderer.addObject(&earth);
    renderer.addObject(&mars);
    renderer.addObject(&jupiter);
    renderer.addObject(&saturn);
    renderer.addObject(&uranus);
    renderer.addObject(&neptune);
    renderer.addObject(&pluto);

    renderer.setAmbientLight({{}, {1.0f, 1.0f, 1.0f}, 1.0f});

    int currentBound = 0;
    std::vector<Planet*> planets = {&sun, &mercury, &venus, &earth, &mars, &jupiter, &saturn, &uranus, &neptune, &pluto};
    auto f = std::bind(bindCamera, planets, &currentBound, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    sun.frameCallbacks.push_back(f);

    renderer.run();
    return 0;
}