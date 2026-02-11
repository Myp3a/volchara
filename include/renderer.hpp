#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <vector>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_beta.h>
#include <vulkan/vulkan_raii.hpp>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <objects.hpp>
#include <raii_wrappers.hpp>


namespace volchara {
    const uint32_t WIDTH = 800;
    const uint32_t HEIGHT = 600;

    const int MAX_FRAMES_IN_FLIGHT = 2;
    const int MAX_FRAMERATE = 60;

    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    struct SwapChainSupportDetails {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;
    };

    enum DebugViewMode {
        OFF,
        NORMALS,
        DEPTH,
        WIREFRAME,
        UNLIT,
    };

    struct DebugFeatures {
        bool culling = true;
        DebugViewMode viewMode = DebugViewMode::OFF;
        bool lightning = true;
    };

    class Renderer {
        friend class volchara::Object;
        friend class volchara::GLTFModel;

        uint32_t maxTextures = 64;

        public:
            Renderer();
            void init();
            void run();
            const std::filesystem::path& getResourceDir();
            void addObject(volchara::Object* obj);
            void delObject(volchara::Object* obj);
            void addLight(volchara::DirectionalLight* obj);
            Plane objPlaneFromWorldCoordinates(InitDataPlane vertices);
            Object objGLTFModelFromFile(std::filesystem::path modelPath);
            Box objBoxFromWorldCoordinates(InitDataBox vertices);
            void setAmbientLight(InitDataLight data);
            DirectionalLight objDirectionalLightFromWorldCoordinates(InitDataLight data);
            void preloadTexture(std::filesystem::path texturePath);
            void preloadModel(std::filesystem::path modelPath);

            static std::vector<unsigned char> readFile(const std::filesystem::path filename, bool asText = false) {
                std::ifstream file(filename, std::ios::ate | (asText ? 0 : std::ios::binary));
        
                if (!file.is_open()) {
                    throw std::runtime_error("failed to open file!");
                }
        
                size_t fileSize = (size_t) file.tellg();
                std::vector<unsigned char> buffer(fileSize);
        
                file.seekg(0);
                file.read(reinterpret_cast<char *>(buffer.data()), fileSize);
        
                file.close();
        
                return buffer;
            }

            volchara::Camera camera;
            bool shouldExit = false;
        private:
            DebugFeatures debugFeatures;
            const std::vector<const char*> validationLayers = {
                "VK_LAYER_KHRONOS_validation"
            };
            #if __APPLE__
            const std::vector<const char*> instanceExtensions = {
                VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
                VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
            };
            const std::vector<const char*> deviceExtensions = {
                VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
                VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
                VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME
                VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,
                VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
            };
            #else
            const std::vector<const char*> instanceExtensions = {
                VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
            };
            const std::vector<const char*> deviceExtensions = {
                VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
                VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
                VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,
                VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
            };
            #endif
            #ifdef NDEBUG
            const bool enableValidationLayers = false;
            #else
            const bool enableValidationLayers = true;
            #endif
            static bool hasRequiredPhysicalDeviceFeatures(vk::PhysicalDeviceFeatures2 deviceFeatures) {
                return deviceFeatures.features.samplerAnisotropy && deviceFeatures.features.fillModeNonSolid;
            }
            static bool hasRequiredPhysicalDeviceDescriptorFeatures(vk::PhysicalDeviceDescriptorIndexingFeaturesEXT deviceFeatures) {
                return deviceFeatures.descriptorBindingPartiallyBound && deviceFeatures.descriptorBindingSampledImageUpdateAfterBind && deviceFeatures.descriptorBindingVariableDescriptorCount && deviceFeatures.runtimeDescriptorArray;
            }
            GLFWwindow* window;
        
            vk::raii::Context context;
            vk::raii::Instance instance = nullptr;
            vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
            vk::raii::SurfaceKHR surface = nullptr;
        
            vk::raii::PhysicalDevice physicalDevice = nullptr;
            vk::PhysicalDeviceProperties physicalDeviceProperties;
            vk::raii::Device device = nullptr;

            DeviceBufferCopyHandler deviceBufferCopyHandler = nullptr;
            RAIIAllocator allocator = nullptr;
        
            vk::raii::Queue graphicsQueue = nullptr;
            vk::raii::Queue presentQueue = nullptr;
        
            vk::raii::SwapchainKHR swapChain = nullptr;
            std::vector<vk::Image> swapChainImages;
            vk::Format swapChainImageFormat;
            vk::Extent2D swapChainExtent;
            std::vector<vk::raii::ImageView> swapChainImageViews;
            std::vector<vk::raii::Framebuffer> swapChainFramebuffers;
        
            vk::raii::RenderPass renderPass = nullptr;
            vk::raii::DescriptorSetLayout descriptorSetLayoutUBO = nullptr;
            vk::raii::DescriptorSetLayout descriptorSetLayoutTextures = nullptr;
            vk::raii::DescriptorSetLayout descriptorSetLayoutSSBO = nullptr;
            vk::raii::DescriptorSetLayout descriptorSetLayoutLightSubpass = nullptr;
            vk::raii::PipelineLayout colorPipelineLayout = nullptr;
            vk::raii::PipelineLayout lightPipelineLayout = nullptr;
            vk::raii::PipelineLayout transparencyPipelineLayout = nullptr;
            vk::raii::Pipeline colorGraphicsPipeline = nullptr;
            vk::raii::Pipeline lightGraphicsPipeline = nullptr;
            vk::raii::Pipeline transparencyGraphicsPipeline = nullptr;
        
            vk::raii::CommandPool commandPool = nullptr;
            std::vector<vk::raii::CommandBuffer> commandBuffers;
        
            std::vector<vk::raii::Semaphore> imageAvailableSemaphores;
            std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
            std::vector<vk::raii::Fence> inFlightFences;
            uint32_t currentFrame = 0;
            std::chrono::time_point<std::chrono::steady_clock> lastFrameTime = std::chrono::steady_clock::now();
        
            RAIIvmaBuffer stagingBuffer = nullptr;
            RAIIvmaBuffer vertexBuffer = nullptr;
            RAIIvmaBuffer indexBuffer = nullptr;
            RAIIvmaBuffer ssboBuffer = nullptr;
            std::vector<RAIIvmaBuffer> uniformBuffers;
            std::vector<RAIIvmaImage> depthBuffers;
            std::vector<RAIIvmaImage> emissiveBuffers;
            std::vector<RAIIvmaImage> normalBuffers;
            std::vector<RAIIvmaImage> intermediateColorBuffers;

            vk::raii::DescriptorPool descriptorPool = nullptr;
            std::vector<vk::raii::DescriptorSet> descriptorSetsUBO;
            std::vector<vk::raii::DescriptorSet> descriptorSetsTextures;
            std::vector<vk::raii::DescriptorSet> descriptorSetsSSBO;
            std::vector<vk::raii::DescriptorSet> descriptorSetsLightSubpass;

            PushConstants pushConstants;

            vk::raii::Sampler textureSampler = nullptr;
            std::vector<RAIIvmaImage> textures;
            std::map<std::string, int> textureNameToId;
            std::map<std::string, tinygltf::Model> modelCache;
        
            std::set<int> pressedKeys;
            glm::vec2 cursorOffset;
            glm::vec2 prevOffset;
            float cameraSpeed = 1.0f;
            float mouseSensitivity = 1.0f;
        
            std::vector<volchara::Object*> objects {};
            volchara::GPULightsBuffer lights {};
        
            bool framebufferResized = false;
        
            static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
            {
                Renderer* app = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));
                if (action == GLFW_PRESS) {
                    app->pressedKeys.insert(key);
                }
                else if (action == GLFW_RELEASE) {
                    app->pressedKeys.erase(key);
                }
            }

            static void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) 
            {
                Renderer* app = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));
                app->cursorOffset.x += xpos - app->prevOffset.x;
                app->cursorOffset.y += ypos - app->prevOffset.y;
                app->prevOffset.x = xpos;
                app->prevOffset.y = ypos;
            }

            static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
                Renderer* app = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));
                app->framebufferResized = true;
            }

            void putObjectsToBuffer();
            void putLightsToBuffer();
            void initWindow();
            void initVulkan();
            void mainLoop();
            void cleanup();
            bool checkValidationLayerSupport();
            std::vector<const char*> getRequiredExtensions();
            void createInstance();
            void setupDebugMessenger();
            void createSurface();
            QueueFamilyIndices findQueueFamilies(vk::raii::PhysicalDevice device);
            bool checkDeviceExtensionSupport(vk::raii::PhysicalDevice device);
            SwapChainSupportDetails querySwapChainSupport(vk::raii::PhysicalDevice device);
            bool isDeviceSuitable(vk::raii::PhysicalDevice device);
            void pickPhysicalDevice();
            void createLogicalDevice();
            void createBufferCopyHandler();
            void createMemoryAllocator();
            void createTextureSampler();
            vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
            vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
            vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
            void createSwapChain();
            vk::raii::ImageView createImageView(const vk::Image& image, vk::Format format);
            void createImageViews();
            vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features);
            vk::Format findDepthFormat();
            void createRenderPass();
            void createDescriptorSetLayout();
            vk::raii::ShaderModule createShaderModule(const std::vector<unsigned char>& code);
            void createGraphicsPipeline();
            void createCommandPool();
            void createStagingBuffer(uint32_t size);
            void createVertexBuffer(uint32_t size);
            void createIndexBuffer(uint32_t size);
            void createUniformBuffers();
            void createSSBOBuffer(uint32_t size);
            RAIIvmaImage createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::ImageAspectFlags aspectFlags = vk::ImageAspectFlagBits::eColor);
            vk::raii::CommandBuffer beginSingleTimeCommands();
            void endSingleTimeCommands(vk::raii::CommandBuffer& buffer);
            void transitionImageLayout(const vk::Image& image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
            void createDepthResources();
            void createEmissiveResources();
            void createNormalResources();
            void createIntermediateColorResources();
            void createFramebuffers();
            uint32_t createTextureImage(std::vector<unsigned char> textureData);
            void createDescriptorPool();
            void createDescriptorSets();
            uint32_t loadTextureToDescriptors(uint32_t textureIndex);
            void createCommandBuffers();
            void createSyncObjects();
            void updateCameraPosition(float passedSeconds);
            void handleDebugModes();
            void recreateSwapChain();
            void recordCommandBuffer(uint32_t imageIndex, uint32_t bufferIndex);
            void updateUniformBuffer(uint32_t imageIndex);
            void drawFrame();
        
            static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageType, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
                std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        
                return VK_FALSE;
            }
        };
}