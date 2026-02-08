# volchara
A custom work-in-progress 3D Vulkan renderer.

# Setup
Clone the repository and open the project, CMake will handle the rest.

# Usage
Not yet available :P  
Nevertheless, you can setup and display a simple scene:
```c++
#include <renderer.hpp>

volchara::Renderer renderer{};

int main() {
    volchara::Box object = renderer.objBoxFromWorldCoordinates({{0.45f, 0.3f, -0.65f}, {1, 2, 1}, {{-0.3f, 0.5f, -0.01f}, {-0.3f, 0.5f, 0.01f}, {-0.3f, -0.5f, -0.01f}}});
    renderer.addObject(&object);
    volchara::DirectionalLight light = renderer.objDirectionalLightFromWorldCoordinates({{2.0f, 2.0f, 2.0f}, {1.0f, 1.0f, 1.0f}, 25.0f});
    renderer.addLight(&light);

    renderer.run();
    return 0;
}
```

# Hotkeys
- `RCtrl + 1` - standard view mode
- `RCtrl + 2` - normal view debug mode
- `RCtrl + 3` - depth view debug mode
- `RCtrl + 4` - wireframe debug mode
- `RCtrl + C` - toggle culling
- `WASDQE + Mouse` - camera control
- `Esc` - exit
