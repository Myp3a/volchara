add_executable(katamari katamari.cpp)

include(../../cmake/compile_shaders.cmake)
include(../../cmake/copy_resources.cmake)

target_link_libraries(katamari PUBLIC volchara)

define_resource_set(
    NAME katamari_textures
    GLOB "textures/*"
)
define_resource_set(
    NAME katamari_models
    GLOB "models/*"
)
use_resource_set(TARGET volchara SETS katamari_textures)
use_resource_set(TARGET volchara SETS katamari_models)
use_resource_set(TARGET volchara SETS base_textures)
use_resource_set(TARGET volchara SETS base_models)
use_shader_set(TARGET volchara SETS base_shaders)
