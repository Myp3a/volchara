#include <chrono>
#include <random>
#include <vector>

#include <tracy/Tracy.hpp>

#include <renderer.hpp>

volchara::Renderer renderer{};
std::mt19937 rnd(std::chrono::time_point<std::chrono::system_clock>().time_since_epoch().count());
std::uniform_real_distribution<> boolDis(-1.0f, 1.0f);
std::uniform_real_distribution<> positionDis(0.5f, 3.0f);
std::uniform_int_distribution<> colorDis(1, 14);

class Cat : public volchara::Object {
    public:
        Cat(volchara::Renderer& renderer) : volchara::Object(volchara::GLTFModel::fromFile(renderer, renderer.getResourceDir() / "models/CatModel.glb")) {
            this->replaceTextures(renderer.getResourceDir() / std::format("textures/Cat_color_{}.png", colorDis(rnd)));
            this->children[0]->transform.rotation.left(180);
        }
};

void placeMoreCats(std::vector<Cat*>* freeCats, volchara::Object* obj, volchara::FrameCallbackData cbData) {
    if (freeCats->size() < 10) {
        Cat* cat = new Cat(renderer);
        if (boolDis(rnd) > 0) {
            cat->transform.position.right(positionDis(rnd));
        } else {
            cat->transform.position.left(positionDis(rnd));
        }
        if (boolDis(rnd) > 0) {
            cat->transform.position.forward(positionDis(rnd));
        } else {
            cat->transform.position.backward(positionDis(rnd));
        }
        freeCats->push_back(cat);
        renderer.addObject(cat);
    }
}

void moveCat(volchara::Object* obj, volchara::FrameCallbackData cbData) {
    float speed = 1.0f;
    if (cbData.pressedKeys.contains(GLFW_KEY_W)) {
        obj->transform.position.forward(cbData.passedSeconds * speed);
        obj->children[0]->transform.rotation.down(cbData.passedSeconds * speed * 180, true);
    }
    if (cbData.pressedKeys.contains(GLFW_KEY_S)) {
        obj->transform.position.backward(cbData.passedSeconds * speed);
        obj->children[0]->transform.rotation.up(cbData.passedSeconds * speed * 180, true);
    }
    if (cbData.pressedKeys.contains(GLFW_KEY_A)) {
        obj->transform.position.left(cbData.passedSeconds * speed);
        obj->children[0]->transform.rotation.ccw(cbData.passedSeconds * speed * 180, true);
    }
    if (cbData.pressedKeys.contains(GLFW_KEY_D)) {
        obj->transform.position.right(cbData.passedSeconds * speed);
        obj->children[0]->transform.rotation.cw(cbData.passedSeconds * speed * 180, true);
    }
    if (cbData.pressedKeys.contains(GLFW_KEY_UP)) {
        obj->children[1]->children[0]->transform.position.forward(cbData.passedSeconds * speed);
    }
    if (cbData.pressedKeys.contains(GLFW_KEY_DOWN)) {
        obj->children[1]->children[0]->transform.position.backward(cbData.passedSeconds * speed);
    }
    obj->children[1]->transform.rotation.right(cbData.cursorOffset.x * cbData.passedSeconds);
}

void addIfCollides(std::vector<Cat*>* freeCats, volchara::Object* obj, volchara::FrameCallbackData cbData) {
    std::vector<volchara::Object*> pile = {obj->children[0]};
    for (int i = 0; i < pile.size(); i++) {
        for (volchara::Object* ptr : pile[i]->children) {
            Cat* catPtr = dynamic_cast<Cat*>(ptr);
            if (catPtr != nullptr) {
                pile.push_back(ptr);
            }
        }
        for (Cat* cat : *freeCats) {
            if (glm::distance(cat->transform.position.world(), pile[i]->transform.position.world()) < 0.25) {
                cat->transform.translation -= obj->transform.translation;
                // TODO: better snapping
                cat->transform.rotationQuat = glm::conjugate(obj->children[0]->transform.rotationQuat);
                obj->children[0]->children.push_back(cat);
                cat->parent = obj->children[0];
                auto catPos = std::find_if(freeCats->begin(), freeCats->end(), [&](Cat* ptr){ return ptr == cat;});
                freeCats->erase(catPos);
                break;
            }
        }
    }
}

int main() {
    for (int i = 1; i < 15; i++) {
        renderer.preloadTexture(renderer.getResourceDir() / std::format("textures/Cat_color_{}.png", i));
    }
    renderer.preloadModel(renderer.getResourceDir() / "models/CatModel.glb");
    Cat mainCat = Cat(renderer);
    mainCat.replaceTextures(renderer.getResourceDir() / "textures/Cat_color_1.png");
    volchara::Object anchor = volchara::Object(renderer, {});
    mainCat.parent = &anchor;
    anchor.children.push_back(&mainCat);
    anchor.frameCallbacks.push_back(moveCat);
    std::vector<Cat*> freeCats;
    auto f1 = std::bind(placeMoreCats, &freeCats, std::placeholders::_1, std::placeholders::_2);
    anchor.frameCallbacks.push_back(f1);

    auto f2 = std::bind(addIfCollides, &freeCats, std::placeholders::_1, std::placeholders::_2);
    anchor.frameCallbacks.push_back(f2);

    volchara::Object cameraAnchor = volchara::Object(renderer, {});
    renderer.camera.parent = &cameraAnchor;
    cameraAnchor.children.push_back(&renderer.camera);
    cameraAnchor.parent = &anchor;
    anchor.children.push_back(&cameraAnchor);
    renderer.camera.transform.position.backward(1.5, true);
    renderer.camera.transform.position.up(1.5, true);
    renderer.camera.transform.rotationQuat = glm::quatLookAtRH(glm::normalize(glm::vec3{0, -1, -1}), {0, 1, 0});
    renderer.addObject(&anchor);

    renderer.setAmbientLight({{0,0,0}, {1, 1, 1}, 1});
    renderer.run();
    return 0;
}
