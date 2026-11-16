# Starting another project

Choose whether the new application will adapt the supplied viewer or consume only the Vulkan core. In either case, keep the dependency revision pinned so a future renderer change is an intentional upgrade.

**To use this repository as a starter**, clone it with submodules and follow the Windows or Android build instructions in the [README](../README.md). Keep the sample working while introducing your application target. Put game/world code in a new directory such as `games/my_game/`, reusable GPU facilities in `engine/`, and operating-system work in `platforms/`.

For a voxel game, block definitions, chunk data, meshing, terrain generation, world streaming, collision, and save data belong to the game target. Use the sample's asynchronous mesh replacement as a reference for uploading rebuilt chunks. Move a feature into the core only when it represents reusable GPU or platform infrastructure with an explicit ownership contract.

Change the application/project names, window title, settings directory, and Android package/application identifier for the new product. The supplied viewer is a reference scene: its cube shaders, camera behavior, debug controls, and assets can be replaced without changing device selection or allocation code. Preserve third-party license notices when redistributing dependencies.

**To consume only the core from another CMake project**, add this repository beneath `external/vulkan_sandbox/`, preferably as a pinned submodule. Initialize its VMA submodule. The shared core requires Vulkan headers/loader and C++20; it does not require GLFW, ImGui, GLM, or Slang when samples are disabled.

```cmake
cmake_minimum_required(VERSION 3.24)
project(my_game LANGUAGES C CXX)

set(SANDBOX_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(SANDBOX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(external/vulkan_sandbox)

add_executable(my_game
    src/main.cpp
    src/my_platform.cpp
    src/my_renderer.cpp
)
target_compile_features(my_game PRIVATE cxx_std_20)
target_link_libraries(my_game PRIVATE sandbox::core)
```

`sandbox::core` is an alias for the source-tree target, not an installed `find_package()` package. Your application provides a `sandbox::Platform` implementation and owns its event/render loop. Use the existing Windows and Android adapters as concrete references for native-window ownership and asset access. The adapter must keep its native window valid until the viewer/device has released the Vulkan surface.

The basic ownership order is:

```cpp
#include <sandbox/core/device.hpp>
#include <sandbox/gpu/resources.hpp>
#include <sandbox/presentation/swapchain.hpp>

// MyPlatform owns a live native window and implements sandbox::Platform.
MyPlatform platform;
sandbox::Device device(platform, true);
sandbox::Allocator allocator(device);
sandbox::UploadQueue uploads(device, allocator);
sandbox::Swapchain swapchain(device, platform, true);

// Create application GPU resources after their allocator.
// Supply frame slots, record Vulkan commands, submit, and present in your loop.
// At shutdown, finish rendering/uploads before resource owners leave scope.
device.wait_idle();
// Swapchain destruction separately waits submitted present fences when enabled.
```

This sketch establishes dependencies; it is not a complete render loop. Cleanup must also cover exceptions, because resource destructors do not wait for work that still uses them. The full sequence is in `samples/viewer/src/viewer.cpp`, including resize handling, per-frame acquisition synchronization, per-image presentation synchronization, optional present fences, and cleanup. Applications that want the supplied scene and loop unchanged can instead enable samples and link `sandbox::viewer`.

**Shader compilation is optional infrastructure.** The helper in `cmake/SandboxShaders.cmake` runs the host's Slang compiler, even for an Android cross-build. A consumer that uses its own shader pipeline does not need to include it. To use the supplied helper:

```cmake
include("${CMAKE_CURRENT_SOURCE_DIR}/external/vulkan_sandbox/cmake/SandboxShaders.cmake")
sandbox_compile_shader(my_game_shaders
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/world.slang"
    "shaders/world.spv")
add_dependencies(my_game my_game_shaders)

add_custom_target(my_game_assets ALL
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${SANDBOX_ASSET_DIRECTORY}"
        "$<TARGET_FILE_DIR:my_game>/assets"
    DEPENDS my_game_shaders)
add_dependencies(my_game my_game_assets)
```

The helper defaults to `vertMain:vertex` and `fragMain:fragment`, SPIR-V 1.5, and column-major matrix layout. Pass `ENTRY_POINTS computeMain:compute` for a compute shader, `PROFILE` to change the SPIR-V profile, and `DEPENDS` for imported shader files outside the source directory. Keep shader and CPU matrix conventions consistent. Set `SLANGC_EXECUTABLE` explicitly when compiler discovery is unsuitable. Generated outputs belong in the build tree; the destination asset path is controlled by `SANDBOX_ASSET_DIRECTORY`. The always-built asset target also copies shader-only edits when the executable does not relink.

The executable copy step above is for desktop packaging. Android must package the generated files into APK assets and read them through `AAssetManager`, as the supplied Android project does. The Android target links the NDK's Vulkan loader, not a Windows SDK library. Shader compilation still runs on the development host.

**Keep resource changes safe while frames overlap.** Destination buffers/images must outlive their upload tickets and every draw that uses them. `UploadQueue` owns only staging data. Use per-frame resources for changing uniforms, and use a new allocation plus deferred replacement when updating a mesh that earlier frames may still read. Serialize upload and draw submissions on the rendering thread. See [the ownership and upload rules](architecture.md) before adding background meshing or streaming.

**Make the new application's requirements explicit.** The delivered renderer requires Vulkan 1.3 with dynamic rendering and synchronization2. The supplied presentation path requires identity surface transform. Enabling Android packaging or choosing an Android SDK level does not establish GPU support. Verify the intended physical devices and document any stricter features your shaders or renderer introduce.

Before using the new application as another template, verify a fresh Debug/Release build, launch from outside the repository, asset loading from the installed package, resize/minimize recovery, and clean shutdown. On Android, additionally exercise pause/resume, screen lock, rotation, surface replacement, touch input, and packaged assets on physical hardware. Keep successful compilation separate from recorded device-runtime results.
