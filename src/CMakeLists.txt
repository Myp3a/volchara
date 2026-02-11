add_library(volchara renderer.cpp objects.cpp raii_wrappers.cpp device_buffer_copy_handler.cpp extlibs/vma/vk_mem_alloc.cpp)
target_include_directories(volchara PUBLIC ../include)

target_compile_definitions(volchara PUBLIC VULKAN_HPP_NO_STRUCT_CONSTRUCTORS PUBLIC GLM_ENABLE_EXPERIMENTAL PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE PUBLIC GLM_FORCE_DEFAULT_ALIGNED_GENTYPES)

include(../cmake/CPM.cmake)
include(../cmake/compile_shaders.cmake)
include(../cmake/copy_resources.cmake)

find_package(Vulkan)
target_link_libraries(volchara PUBLIC Vulkan::Vulkan)

CPMAddPackage("gh:GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator#v3.2.1")
target_link_libraries(volchara PUBLIC VulkanMemoryAllocator)
CPMAddPackage("gh:YaaZ/VulkanMemoryAllocator-Hpp#v3.2.1")
target_link_libraries(volchara PUBLIC VulkanMemoryAllocator-Hpp)

CPMAddPackage("gh:glfw/glfw#3.4")
target_link_libraries(volchara PUBLIC glfw)

CPMAddPackage("gh:g-truc/glm#1.0.1")
target_link_libraries(volchara PUBLIC glm)

CPMAddPackage("gh:wolfpld/tracy#v0.13.1")
target_link_libraries(volchara PUBLIC Tracy::TracyClient)

add_library(stb_image STATIC extlibs/stb_image/stb_image.cpp)
target_include_directories(stb_image PUBLIC extlibs/stb_image)
target_link_libraries(volchara PUBLIC stb_image)

add_library(tinygltf STATIC extlibs/tinygltf/tiny_gltf.cpp)
target_include_directories(tinygltf PUBLIC extlibs/tinygltf PUBLIC extlibs/stb_image)
target_link_libraries(volchara PUBLIC tinygltf)

define_shader_set(
    NAME base_shaders
    GLOB "../shaders/*.frag" "../shaders/*.vert"
)
use_shader_set(TARGET volchara SETS base_shaders)

define_resource_set(
    NAME base_textures
    GLOB "../textures/*"
    ROOT "../"
)
use_resource_set(TARGET volchara SETS base_textures)

define_resource_set(
    NAME base_models
    GLOB "../models/*"
    ROOT "../"
)
use_resource_set(TARGET volchara SETS base_models)

set(RESOURCE_DIR "${CMAKE_BINARY_DIR}/resources/")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/resource_path.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/resource_path.hpp
    @ONLY
)
target_include_directories(volchara PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
if (MSVC)
    target_compile_options(volchara PRIVATE /utf-8)
    target_compile_definitions(volchara PRIVATE UNICODE _UNICODE)
endif()
