#include <array>
#include <filesystem>
#include <numeric>
#include <set>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan_raii.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/hash.hpp>
#include <stb_image.h>
#include <tiny_gltf.h>

#include <objects.hpp>
#include <renderer.hpp>


namespace volchara {
    template<class T>
    inline void hash_combine(std::size_t& seed, T const& v) noexcept {
        seed ^= std::hash<T>{}(v) + 0x9e3779b97f4a7c15ull + (seed<<6) + (seed>>2);
    }

    struct VertexHash {
        std::size_t operator()(Vertex const& v) const {
            std::size_t seed = 0;
            hash_combine(seed, v.pos);
            hash_combine(seed, v.normal);
            hash_combine(seed, v.color);
            hash_combine(seed, v.texCoord);
            return seed;
        }
    };

    bool Vertex::operator==(const Vertex& other) const {
        return ((pos == other.pos) && (color == other.color) && (texCoord == other.texCoord));
    }

    vk::VertexInputBindingDescription Vertex::getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex,
        };

        return bindingDescription;
    }

    std::vector<vk::VertexInputAttributeDescription> Vertex::getAttributeDescriptions() {
        std::vector<vk::VertexInputAttributeDescription> attributeDescriptions{};

        vk::VertexInputAttributeDescription positionDescription{
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(Vertex, pos),
        };
        attributeDescriptions.push_back(positionDescription);

        vk::VertexInputAttributeDescription normalDescription{
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(Vertex, normal),
        };
        attributeDescriptions.push_back(normalDescription);

        vk::VertexInputAttributeDescription colorDescription{
            .location = 2,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(Vertex, color),
        };
        attributeDescriptions.push_back(colorDescription);

        vk::VertexInputAttributeDescription textureCoordinatesDescription{
            .location = 3,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(Vertex, texCoord),
        };
        attributeDescriptions.push_back(textureCoordinatesDescription);
    
        return attributeDescriptions;
    }

    void Transform::Position::forward(float distance, bool world) {
        if (world) {
            parent->translation += glm::vec3(0, 0, -distance);
        } else {
            parent->translation += parent->rotationQuat * glm::vec3(0, 0, -distance);
        }
    }
    void Transform::Position::backward(float distance, bool world) {
        forward(-distance, world);
    }
    void Transform::Position::left(float distance, bool world) {
        if (world) {
            parent->translation += glm::vec3(-distance, 0, 0);
        } else {
            parent->translation += parent->rotationQuat * glm::vec3(-distance, 0, 0);
        }
    }
    void Transform::Position::right(float distance, bool world) {
        left(-distance, world);
    }
    void Transform::Position::up(float distance, bool world) {
        if (world) {
            parent->translation += glm::vec3(0, distance, 0);
        } else {
            parent->translation += parent->rotationQuat * glm::vec3(0, distance, 0);
        }
    }
    void Transform::Position::down(float distance, bool world) {
        up(-distance, world);
    }

    void Transform::Rotation::up(float degrees, bool world) {
        if (world) {
            parent->rotationQuat = glm::normalize(glm::quat({degrees, 0, 0}) * parent->rotationQuat);
        } else {
            parent->rotationQuat = glm::normalize(parent->rotationQuat * glm::quat({degrees, 0, 0}));
        }
    }
    void Transform::Rotation::down(float degrees, bool world) {
        up(-degrees, world);
    }
    void Transform::Rotation::left(float degrees, bool world) {
        if (world) {
            parent->rotationQuat = glm::normalize(glm::quat({0, degrees, 0}) * parent->rotationQuat);
        } else {
            parent->rotationQuat = glm::normalize(parent->rotationQuat * glm::quat({0, degrees, 0}));
        }
    }
    void Transform::Rotation::right(float degrees, bool world) {
        left(-degrees, world);
    }
    void Transform::Rotation::cw(float degrees, bool world) {
        if (world) {
            parent->rotationQuat = glm::normalize(glm::quat({0, 0, -degrees}) * parent->rotationQuat);
        } else {
            parent->rotationQuat = glm::normalize(parent->rotationQuat * glm::quat({0, 0, -degrees}));
        }
    }
    void Transform::Rotation::ccw(float degrees, bool world) {
        cw(-degrees, world);
    }

    glm::mat4 Transform::modelMatrix() {
        glm::mat4 translationMatrix = glm::translate(translation);
        glm::mat4 scaleMatrix = glm::scale(scaling);
        glm::mat4 rotationMatrix = glm::toMat4(rotationQuat);
        glm::mat4 result = translationMatrix * rotationMatrix * scaleMatrix;
        return result;
    }

    Object::Object(Renderer& renderer, std::vector<Vertex> initVertices, std::vector<uint32_t> initIndices, glm::vec3 translation, glm::vec3 scaling, glm::quat rotation) {
        this->renderer = &renderer;
        vertices = initVertices;
        if (initIndices.empty()) {
            indices = std::vector<uint32_t>(vertices.size());
            std::iota(indices.begin(), indices.end(), 0);
        }
        else {
            indices = initIndices;
        }
        transform.translation = translation;
        transform.scaling = scaling;
        transform.rotationQuat = rotation;
    }
    void Object::runFrameCallbacks(float passedSeconds, std::set<int> pressedKeys) {
        for (auto callback : frameCallbacks) {
            callback(this, passedSeconds, pressedKeys);
        }
        return;
    }
    void Object::setColor(std::array<float, 3> color) {
        for (Vertex& v : vertices) {
            v.color.r = color[0];
            v.color.g = color[1];
            v.color.b = color[2];
        }
    }
    void Object::loadTexture(const std::filesystem::path path) {
        textureIndex = renderer->createTextureImage(path);
        renderer->loadTextureToDescriptors(textureIndex);
    }
    void Object::generateIndices(std::vector<Vertex> fromVertices) {
        std::vector<Vertex> newVertices;
        std::vector<uint32_t> newIndices;
        std::unordered_map<Vertex, int32_t, VertexHash> indexMap;
        for (Vertex v : fromVertices) {
            auto pos = indexMap.find(v);
            if (pos == indexMap.end()) {
                newIndices.push_back(newVertices.size());
                indexMap.insert({v, newVertices.size()});
                newVertices.push_back(v);
            } else {
                newIndices.push_back(pos->second);
            }
        }
        maxVertexIndex = *std::max_element(newIndices.begin(), newIndices.end());
        vertices = newVertices;
        indices = newIndices;
    }

    Plane Plane::fromWorldCoordinates(Renderer& renderer, InitDataPlane initVertices, bool wIndices) {
        std::vector<Vertex> vertices;
        glm::vec3 topLeft = {initVertices.topLeft[0], initVertices.topLeft[1], initVertices.topLeft[2]};
        glm::vec3 topRight = {initVertices.topRight[0], initVertices.topRight[1], initVertices.topRight[2]};
        glm::vec3 botRight = {initVertices.botRight[0], initVertices.botRight[1], initVertices.botRight[2]};
        glm::vec3 botLeft = topLeft - (topRight - botRight);
        glm::vec3 x = topRight - topLeft;
        glm::vec3 y = topLeft - botLeft;
        glm::vec3 z = glm::normalize(glm::cross(x, y));
        glm::vec3 center = botLeft + x / 2.0f + y / 2.0f;
        float width = glm::length(x);
        float height = glm::length(y);
        // Plane has its texture on its front, so UVs are hacky to make it work
        vertices.push_back({.pos = {-width / 2.0f, height / 2.0f, 0}, .normal = z, .texCoord = {1, 0}});
        vertices.push_back({.pos = {width / 2.0f, height / 2.0f, 0}, .normal = z, .texCoord = {0, 0}});
        vertices.push_back({.pos = {width / 2.0f, -height / 2.0f, 0}, .normal = z, .texCoord = {0, 1}});
        vertices.push_back({.pos = {-width / 2.0f, height / 2.0f, 0}, .normal = z, .texCoord = {1, 0}});
        vertices.push_back({.pos = {width / 2.0f, -height / 2.0f, 0}, .normal = z, .texCoord = {0, 1}});
        vertices.push_back({.pos = {-width / 2.0f, -height / 2.0f, 0}, .normal = z, .texCoord = {1, 1}});
        Plane obj(renderer, {}, {}, center, {1, 1, 1}, glm::quatLookAtRH(z, y));
        if (wIndices) {
            obj.generateIndices(vertices);
        }
        return obj;
    }
    
    GLTFModel GLTFModel::fromFile(Renderer &renderer, std::filesystem::path modelPath) {
        tinygltf::TinyGLTF gltfLoader;
        tinygltf::Model model;
        std::string err;
        std::string warn;
        std::u8string unicodePathTmp = modelPath.u8string();
        std::string unicodePath(unicodePathTmp.begin(), unicodePathTmp.end());
        bool res;
        if (modelPath.extension().string() == ".gltf") {
            res = gltfLoader.LoadASCIIFromFile(&model, &err, &warn, unicodePath);
        }
        else if (modelPath.extension().string() == ".glb") {
            res = gltfLoader.LoadBinaryFromFile(&model, &err, &warn, unicodePath);
        }
        else {
            throw std::runtime_error(std::string("failed to load gltf: unknown extension ") + modelPath.extension().string());
        }
        if (!res || !err.empty()) {
            throw std::runtime_error("failed to load gltf: " + err);
        }
        bool solid_color;
        if (model.textures.size() < 1) {
            solid_color = true;
        }
        else {
            solid_color = false;
        }
        std::map<int, int> textureMapping;
        for (tinygltf::Texture& texture : model.textures) {
            int modelTextureId = texture.source;
            std::filesystem::path texturePath = modelPath.parent_path() / model.images[modelTextureId].uri;
            int rendererTextureId = renderer.createTextureImage(texturePath);
            renderer.loadTextureToDescriptors(rendererTextureId);
            textureMapping[modelTextureId] = rendererTextureId;
        }
        std::vector<Vertex> resVertices;
        std::vector<uint32_t> resIndices;
        tinygltf::Scene& defScene = model.scenes[model.defaultScene];
        std::vector<int> nodes(defScene.nodes.begin(), defScene.nodes.end());
        while (!nodes.empty()) {
            int node_id = nodes.back();
            nodes.pop_back();
            tinygltf::Node& node = model.nodes[node_id];
            for (int child : node.children) {
                nodes.push_back(child);
            }
            if (node.mesh < 0) {
                continue;
            }

            tinygltf::Mesh& mesh = model.meshes[node.mesh];
            for (const tinygltf::Primitive prim : mesh.primitives) {
                if (prim.mode != TINYGLTF_MODE_TRIANGLES && prim.mode != 0) {
                    throw std::runtime_error("failed to load gltf: currently only triangle load available");
                }

                auto iterPosition = prim.attributes.find("POSITION");
                if (iterPosition == prim.attributes.end()) {
                    continue;
                }
                tinygltf::Accessor accessorPosition = model.accessors[iterPosition->second];
                if (accessorPosition.type != TINYGLTF_TYPE_VEC3 || accessorPosition.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
                    throw std::runtime_error("failed to load gltf: position not vec3 float");
                }
                auto iterTexCoord = prim.attributes.find("TEXCOORD_0");
                if (iterTexCoord == prim.attributes.end()) {
                    continue;
                }
                tinygltf::Accessor accessorTexCoord = model.accessors[iterTexCoord->second];
                if (accessorTexCoord.type != TINYGLTF_TYPE_VEC2 || accessorTexCoord.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
                    throw std::runtime_error("failed to load gltf: uv not vec2 float");
                }
                
                tinygltf::BufferView& bufferViewPosition = model.bufferViews[accessorPosition.bufferView];
                tinygltf::Buffer& bufferPosition = model.buffers[bufferViewPosition.buffer];
                const float* positions = reinterpret_cast<const float*>(&bufferPosition.data[bufferViewPosition.byteOffset + accessorPosition.byteOffset]);

                tinygltf::BufferView& bufferViewTexCoord = model.bufferViews[accessorTexCoord.bufferView];
                tinygltf::Buffer& bufferTexCoord = model.buffers[bufferViewTexCoord.buffer];
                const float* texcoords = reinterpret_cast<const float*>(&bufferTexCoord.data[bufferViewTexCoord.byteOffset + accessorTexCoord.byteOffset]);

                if (prim.indices >= 0) {
                    int indexOffset = resVertices.size();
                    tinygltf::Accessor accessorIndices = model.accessors[prim.indices];
                    tinygltf::BufferView& bufferViewIndices = model.bufferViews[accessorIndices.bufferView];
                    tinygltf::Buffer& bufferIndices = model.buffers[bufferViewIndices.buffer];
                    const uint32_t* indices = reinterpret_cast<const uint32_t*>(&bufferIndices.data[bufferViewIndices.byteOffset + accessorIndices.byteOffset]);
                    for (size_t i = 0; i < accessorIndices.count; i++) {
                        uint32_t index = indices[i];
                        resIndices.push_back(index + indexOffset);
                    }
                    for (size_t vertexId = 0; vertexId < accessorPosition.count; vertexId++) {
                        Vertex v;
                        v.pos = glm::vec3(positions[vertexId*3 + 0], positions[vertexId*3 + 1], positions[vertexId*3 + 2]);
                        v.texCoord = glm::vec2(texcoords[vertexId*2 + 0], texcoords[vertexId*2 + 1]);
                        if (solid_color) {
                            v.color = glm::vec3(1, 0, 0);
                        }
                        resVertices.push_back(v);
                    }
                }
                else {
                    for (size_t i = 0; i < accessorPosition.count; i++) {
                        Vertex v;
                        v.pos = glm::vec3(positions[i*3 + 0], positions[i*3 + 1], positions[i*3 + 2]);
                        v.texCoord = glm::vec2(texcoords[i*2 + 0], texcoords[i*2 + 1]);
                        if (solid_color) {
                            v.color = glm::vec3(1, 0, 0);
                        }
                        resIndices.push_back(resVertices.size());
                        resVertices.push_back(v);
                    }
                }
            }
        }
        GLTFModel obj(renderer, resVertices, resIndices);
        obj.textureIndex = textureMapping[0];
        return obj;
    }

    std::array<glm::vec3, 3> Box::calcOrientation(InitDataPlane frontOrientationPlane) {
        glm::vec3 topLeft = {frontOrientationPlane.topLeft[0], frontOrientationPlane.topLeft[1], frontOrientationPlane.topLeft[2]};
        glm::vec3 topRight = {frontOrientationPlane.topRight[0], frontOrientationPlane.topRight[1], frontOrientationPlane.topRight[2]};
        glm::vec3 botRight = {frontOrientationPlane.botRight[0], frontOrientationPlane.botRight[1], frontOrientationPlane.botRight[2]};
        glm::vec3 botLeft = topLeft - (topRight - botRight);
        glm::vec3 x = glm::normalize(topRight - topLeft);
        glm::vec3 y = glm::normalize(topLeft - botLeft);
        glm::vec3 z = glm::normalize(glm::cross(x, y));
        return {x, y, z};
    }

    Box Box::fromWorldCoordinates(Renderer& renderer, InitDataBox initData, bool wIndices) {
        std::vector<Vertex> vertices;
        glm::vec3 center({initData.center[0], initData.center[1], initData.center[2]});
        auto[width, height, depth] = initData.sizes;
        auto[x, y, z] = calcOrientation(initData.frontOrientationPlane);
        // front
        vertices.push_back({.pos = {-width / 2.0f, height / 2.0f, depth / 2.0f}, .normal = {0, 0, 1}, .texCoord = {0, 0}});
        vertices.push_back({.pos = {-width / 2.0f, -height / 2.0f, depth / 2.0f}, .normal = {0, 0, 1}, .texCoord = {0, 1}});
        vertices.push_back({.pos = {width / 2.0f, -height / 2.0f, depth / 2.0f}, .normal = {0, 0, 1}, .texCoord = {1, 1}});
        vertices.push_back({.pos = {-width / 2.0f, height / 2.0f, depth / 2.0f}, .normal = {0, 0, 1}, .texCoord = {0, 0}});
        vertices.push_back({.pos = {width / 2.0f, -height / 2.0f, depth / 2.0f}, .normal = {0, 0, 1}, .texCoord = {1, 1}});
        vertices.push_back({.pos = {width / 2.0f, height / 2.0f, depth / 2.0f}, .normal = {0, 0, 1}, .texCoord = {1, 0}});
        // right
        vertices.push_back({.pos = {width / 2.0f, height / 2.0f, depth / 2.0f}, .normal = {1, 0, 0}, .texCoord = {0, 0}});
        vertices.push_back({.pos = {width / 2.0f, -height / 2.0f, depth / 2.0f}, .normal = {1, 0, 0}, .texCoord = {0, 1}});
        vertices.push_back({.pos = {width / 2.0f, -height / 2.0f, -depth / 2.0f}, .normal = {1, 0, 0}, .texCoord = {1, 1}});
        vertices.push_back({.pos = {width / 2.0f, height / 2.0f, depth / 2.0f}, .normal = {1, 0, 0}, .texCoord = {0, 0}});
        vertices.push_back({.pos = {width / 2.0f, -height / 2.0f, -depth / 2.0f}, .normal = {1, 0, 0}, .texCoord = {1, 1}});
        vertices.push_back({.pos = {width / 2.0f, height / 2.0f, -depth / 2.0f}, .normal = {1, 0, 0}, .texCoord = {1, 0}});
        // back
        vertices.push_back({.pos = {width / 2.0f, height / 2.0f, -depth / 2.0f}, .normal = {0, 0, -1}, .texCoord = {0, 0}});
        vertices.push_back({.pos = {width / 2.0f, -height / 2.0f, -depth / 2.0f}, .normal = {0, 0, -1}, .texCoord = {0, 1}});
        vertices.push_back({.pos = {-width / 2.0f, -height / 2.0f, -depth / 2.0f}, .normal = {0, 0, -1}, .texCoord = {1, 1}});
        vertices.push_back({.pos = {width / 2.0f, height / 2.0f, -depth / 2.0f}, .normal = {0, 0, -1}, .texCoord = {0, 0}});
        vertices.push_back({.pos = {-width / 2.0f, -height / 2.0f, -depth / 2.0f}, .normal = {0, 0, -1}, .texCoord = {1, 1}});
        vertices.push_back({.pos = {-width / 2.0f, height / 2.0f, -depth / 2.0f}, .normal = {0, 0, -1}, .texCoord = {1, 0}});
        // left
        vertices.push_back({.pos = {-width / 2.0f, height / 2.0f, -depth / 2.0f}, .normal = {-1, 0, 0}, .texCoord = {0, 0}});
        vertices.push_back({.pos = {-width / 2.0f, -height / 2.0f, -depth / 2.0f}, .normal = {-1, 0, 0}, .texCoord = {0, 1}});
        vertices.push_back({.pos = {-width / 2.0f, -height / 2.0f, depth / 2.0f}, .normal = {-1, 0, 0}, .texCoord = {1, 1}});
        vertices.push_back({.pos = {-width / 2.0f, height / 2.0f, -depth / 2.0f}, .normal = {-1, 0, 0}, .texCoord = {0, 0}});
        vertices.push_back({.pos = {-width / 2.0f, -height / 2.0f, depth / 2.0f}, .normal = {-1, 0, 0}, .texCoord = {1, 1}});
        vertices.push_back({.pos = {-width / 2.0f, height / 2.0f, depth / 2.0f}, .normal = {-1, 0, 0}, .texCoord = {1, 0}});
        // top
        vertices.push_back({.pos = {width / 2.0f, height / 2.0f, depth / 2.0f}, .normal = {0, 1, 0}, .texCoord = {0, 0}});
        vertices.push_back({.pos = {width / 2.0f, height / 2.0f, -depth / 2.0f}, .normal = {0, 1, 0}, .texCoord = {0, 1}});
        vertices.push_back({.pos = {-width / 2.0f, height / 2.0f, -depth / 2.0f}, .normal = {0, 1, 0}, .texCoord = {1, 1}});
        vertices.push_back({.pos = {width / 2.0f, height / 2.0f, depth / 2.0f}, .normal = {0, 1, 0}, .texCoord = {0, 0}});
        vertices.push_back({.pos = {-width / 2.0f, height / 2.0f, -depth / 2.0f}, .normal = {0, 1, 0}, .texCoord = {1, 1}});
        vertices.push_back({.pos = {-width / 2.0f, height / 2.0f, depth / 2.0f}, .normal = {0, 1, 0}, .texCoord = {1, 0}});
        // bot
        vertices.push_back({.pos = {width / 2.0f, -height / 2.0f, -depth / 2.0f}, .normal = {0, -1, 0}, .texCoord = {0, 0}});
        vertices.push_back({.pos = {width / 2.0f, -height / 2.0f, depth / 2.0f}, .normal = {0, -1, 0}, .texCoord = {0, 1}});
        vertices.push_back({.pos = {-width / 2.0f, -height / 2.0f, depth / 2.0f}, .normal = {0, -1, 0}, .texCoord = {1, 1}});
        vertices.push_back({.pos = {width / 2.0f, -height / 2.0f, -depth / 2.0f}, .normal = {0, -1, 0}, .texCoord = {0, 0}});
        vertices.push_back({.pos = {-width / 2.0f, -height / 2.0f, depth / 2.0f}, .normal = {0, -1, 0}, .texCoord = {1, 1}});
        vertices.push_back({.pos = {-width / 2.0f, -height / 2.0f, -depth / 2.0f}, .normal = {0, -1, 0}, .texCoord = {1, 0}});

        Box obj(renderer, {}, {}, center, {1, 1, 1}, glm::quatLookAtRH(z, y));
        if (wIndices){
            obj.generateIndices(vertices);
        }
        return obj;
    }

    AmbientLight AmbientLight::fromData(Renderer& renderer, InitDataLight initData) {
        glm::vec3 color({initData.color[0], initData.color[1], initData.color[2]});
        AmbientLight light(renderer);
        light.brightness = initData.brightness;
        light.color = color;
        return light;
    }

    DirectionalLight DirectionalLight::fromWorldCoordinates(Renderer& renderer, InitDataLight initData) {
        glm::vec3 position({initData.position[0], initData.position[1], initData.position[2]});
        glm::vec3 color({initData.color[0], initData.color[1], initData.color[2]});
        DirectionalLight light(renderer);
        light.brightness = initData.brightness;
        light.color = color;
        light.transform.translation = position;
        return light;
    }
}