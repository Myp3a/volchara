#pragma once

#include <array>
#include <filesystem>
#include <functional>
#include <vector>
#include <set>

#include <vulkan/vulkan_raii.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <tiny_gltf.h>

namespace volchara {
    class Renderer;

    struct InitDataPlane {
        std::array<float, 3> topLeft;
        std::array<float, 3> topRight;
        std::array<float, 3> botRight;
    };

    struct InitDataBox {
        std::array<float, 3> center;
        std::array<float, 3> sizes;
        InitDataPlane frontOrientationPlane;
    };

    struct InitDataLight {
        std::array<float, 3> position;
        std::array<float, 3> color;
        float brightness;
    };

    struct UniformBufferObject {
        glm::mat4 view;
        glm::mat4 proj;
    };

    struct AmbientLightUniformBufferObject {
        glm::vec3 color;
        float brightness;
    };

    struct DirectionalLightUniformBufferObject : AmbientLightUniformBufferObject {
        glm::mat4 model;
    };

    enum PushConstantsDebugFlags : uint32_t {
        COLOR_NORMALS = 1u << 0,
        COLOR_DEPTH = 1u << 1,
        COLOR_WIREFRAME = 1u << 2,
        COLOR_UNLIT = 1u << 3,
    };

    struct alignas(16) PushConstants {
        glm::mat4 model;
        uint32_t textureIndex = 0;
        uint32_t normalIndex = 0;
        uint32_t emissiveIndex = 0;
        float alphaCutoff = 0.0;
        uint32_t debugFlags = 0;
    };

    struct alignas(16) GPULight {
        glm::vec4 position;
        glm::vec4 color;
    };

    struct alignas(16) GPULightHeader {
        glm::vec4 ambient;
        uint32_t lightCount;
        glm::vec3 pad;
    };

    struct alignas(16) GPULightsBuffer {
        GPULightHeader header;
        GPULight lights[32];
    };

    struct Vertex {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec3 color{0, 0, 0};
        glm::vec2 texCoord;

        bool operator==(const Vertex& other) const;
        static vk::VertexInputBindingDescription getBindingDescription();
        static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions();
    };

    class Object;

    class Transform {
        public:
        Object* parent = nullptr;
        glm::vec3 translation{0,0,0};
        glm::vec3 scaling{1,1,1};
        glm::quat rotationQuat{1,0,0,0};

        class Position {
            private:
            Transform* parent;
            public:
            Position(Transform* loc): parent(loc) {}
            void forward(float distance, bool world = false);
            void backward(float distance, bool world = false);
            void left(float distance, bool world = false);
            void right(float distance, bool world = false);
            void up(float distance, bool world = false);
            void down(float distance, bool world = false);
        };
        Position position = nullptr;

        class Rotation {
            private:
            Transform* parent;
            public:
            Rotation(Transform* loc): parent(loc) {}
            void up(float degrees, bool world = false);
            void down(float degrees, bool world = false);
            void left(float degrees, bool world = false);
            void right(float degrees, bool world = false);
            void ccw(float degrees, bool world = false);
            void cw(float degrees, bool world = false);
        };
        Rotation rotation = nullptr;

        Transform(Object* newParent) {
            parent = newParent;
            position = Position(this);
            rotation = Rotation(this);
        }

        glm::mat4 modelMatrix();
    };

    class CameraTransform : public Transform {
        public:
            glm::mat4 modelMatrix();
    };

    class Object {
    public:
        Object* parent = nullptr;
        std::vector<Object*> children;
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        std::vector<std::function<void(Object*, float, std::set<int>)>> frameCallbacks{};
        Transform transform = Transform(this);
        Renderer* renderer;
        uint32_t textureIndex = 0;
        uint32_t normalIndex = 0;
        uint32_t emissiveIndex = 0;
        float alphaCutoff = 0.0;
        bool transparent = false;

        Object(Renderer &renderer, std::vector<Vertex> initVertices, std::vector<uint32_t> initIndices = {}, glm::vec3 translation = {0, 0, 0}, glm::vec3 scaling = {1, 1, 1}, glm::quat rotation = {1,0,0,0});
        virtual ~Object() = default;  // for RTTI and callback polymorphism
        void runFrameCallbacks(float passedSeconds, std::set<int> pressedKeys);
        void setColor(std::array<float, 3> color);
        void loadTexture(const std::filesystem::path path);
        void generateIndices(std::vector<Vertex> fromVertices);
    };

    class Camera : public Object {
        public:
            Camera(Renderer& renderer) : Object(renderer, {}) {
                transform.rotationQuat = glm::toQuat(glm::lookAt(glm::vec3{0, 0, 1}, {0, 0, 0}, {0, 1, 0}));
            }
    };

    class Plane : public Object {
        public:
            static Plane fromWorldCoordinates(Renderer& renderer, InitDataPlane initVertices, bool wIndices = true);
            Plane(Renderer& renderer, std::vector<Vertex> vertices, std::vector<uint32_t> indices = {}, glm::vec3 translation = {0, 0, 0}, glm::vec3 scaling = {1, 1, 1}, glm::quat rotation = {1,0,0,0}) : Object(renderer, vertices, indices, translation, scaling, rotation) {};
    };

    class GLTFModel : public Object {
        private:
            static Object* traverseNode(Renderer &renderer, tinygltf::Model& model, int nodeId, std::map<int, int>& textureMapping);
        public:
            static Object fromFile(Renderer& renderer, std::filesystem::path modelPath);
            GLTFModel(Renderer& renderer, std::vector<Vertex> vertices, std::vector<uint32_t> indices = {}, glm::vec3 translation = {0, 0, 0}, glm::vec3 scaling = {1, 1, 1}, glm::quat rotation = {1,0,0,0}) : Object(renderer, vertices, indices, translation, scaling, rotation) {};
    };

    class Box : public Object {
        private:
            static std::array<glm::vec3, 3> calcOrientation(InitDataPlane frontOrientationPlane);
        public:
            static Box fromWorldCoordinates(Renderer& renderer, InitDataBox initVertices, bool wIndices = true);
            Box(Renderer& renderer, std::vector<Vertex> vertices, std::vector<uint32_t> indices = {}, glm::vec3 translation = {0, 0, 0}, glm::vec3 scaling = {1, 1, 1}, glm::quat rotation = {1,0,0,0}) : Object(renderer, vertices, indices, translation, scaling, rotation) {};
    };

    class AmbientLight : public Object {
        public:
            float brightness;
            glm::vec3 color{0.0f, 0.0f, 0.0f};
            static AmbientLight fromData(Renderer& renderer, InitDataLight initData);
            AmbientLight(Renderer& renderer) : Object(renderer, {}) {};
    };

    class DirectionalLight : public AmbientLight {
        public:
            glm::vec3 position;
            static DirectionalLight fromWorldCoordinates(Renderer& renderer, InitDataLight initData);
            DirectionalLight(Renderer& renderer) : AmbientLight(renderer) {};
    };
}