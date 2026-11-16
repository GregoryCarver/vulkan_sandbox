# Vulkan Sandbox

A modular C++20 / Vulkan 1.3 starting point for Windows and Android. The shared reference scene renders textured, depth-tested cubes with a fly camera, optional Dear ImGui, and asynchronous mesh replacement. Windows uses GLFW; Android uses GameActivity. Linux/macOS adapters are future work.

## Layout

```text
engine/                 Reusable device, presentation, allocation and upload code
  include/sandbox/      Public core/, gpu/ and presentation/ interfaces
  src/                  Implementations, including the single VMA implementation
samples/viewer/         Shared frame loop, reference scene, camera and debug UI
platforms/windows/      GLFW adapter and desktop executable
platforms/android/      GameActivity adapter, Gradle project and APK packaging
assets/shaders/         Slang source (compiled assets go into the build directory)
cmake/                  Reusable host shader compilation helper
tests/                  CPU regression checks
docs/                   Architecture, Android setup, reuse and validation evidence
third_party/            Pinned Git submodules
```

The core has no GLFW, Android JNI, ImGui, or GLM dependency. It exposes Vulkan directly through a few explicit resource owners. Camera behavior, sample geometry and UI stay in the sample. See [architecture](docs/architecture.md) and [starting a new project](docs/new-project.md).

## Windows

Install Visual Studio 2022 with Desktop development with C++, CMake 3.24+, and a Vulkan SDK containing `slangc`. The tested SDK is 1.4.313.2. Set `VULKAN_SDK`, put `slangc` on PATH, or configure `-DSLANGC_EXECUTABLE=C:/path/to/slangc.exe`.

```powershell
git clone --recurse-submodules https://github.com/GregoryCarver/vulkan_sandbox.git
cd vulkan_sandbox
cmake --preset windows
cmake --build --preset windows-debug --parallel
ctest --preset windows-debug
.\out\windows\Debug\sandbox_windows.exe
```

For an existing checkout, run `git submodule sync --recursive` and `git submodule update --init --recursive`. Keep pinned revisions; `git submodule update --remote` is an intentional dependency upgrade, not a setup step.

Release: `cmake --build --preset windows-release --parallel`. To compile out all ImGui code, configure `cmake --preset windows-no-ui`, then build `cmake --build --preset windows-no-ui --parallel`.

The executable reads assets beside itself and can launch from any working directory. Generated shader changes are copied even when the executable does not relink. Install a self-contained application directory with:

```powershell
cmake --install out/windows --config Release --prefix out/package/windows
```

A Vulkan-capable GPU driver is still required on the destination computer. Release does not require the development SDK; validation is enabled when the layer is available. User UI settings go to `%LOCALAPPDATA%/VulkanSandbox/`, separate from installed assets.

## Android

The first hardware target is the Galaxy S24. The initial renderer requires Vulkan 1.3, dynamic rendering and synchronization2; the app checks capabilities at startup. The Android OS version alone is not a GPU compatibility guarantee. The sample currently uses identity surface transform, with compositor-managed rotation.

Open `platforms/android/` in Android Studio and follow [Android setup](docs/android.md) for the pinned JDK/Gradle/NDK versions, shader compiler configuration, APK build and device checks. Host shader compilation and target NDK compilation are separate; Windows SDK libraries never enter the APK.

**Validation status:** Windows builds/runtime are exercised locally. Android APK compilation and physical Galaxy S24 validation remain pending until the Android toolchain/dependencies are available. See [recorded checks](docs/validation.md) for precise coverage and limitations.

## Controls and diagnostics

WASD moves the camera, Q/E moves down/up, right-drag looks around, and F1 toggles UI. The ImGui demo is available at startup and through the sample panel. Android reserves the bottom 40% of the screen for gestures: drag left to move and right to look. Use the upper area for UI.

```powershell
# Automated resize/minimize/restore and streaming exercise
.\out\windows\Debug\sandbox_windows.exe --hidden --smoke-test --stress --frames 240
# Lower scene resolution while preserving native-resolution UI
.\out\windows\Debug\sandbox_windows.exe --render-scale 0.75
# Exercise the compatibility path without presentation-maintenance fences
.\out\windows\Debug\sandbox_windows.exe --hidden --smoke-test --legacy-present
```

Other options: `--fps N` (`0` uncapped), `--no-vsync`, `--no-ui`, `--no-validation`, `--assets DIR`, and `--frames N`. A failed Vulkan operation or a validation error causes a nonzero smoke-test exit status. Normal shutdown includes teardown validation in the reported count.

## Reuse

Set `SANDBOX_BUILD_SAMPLES=OFF` to consume only `sandbox::core` from another CMake project. This excludes GLFW, ImGui, GLM and Slang from the build requirements. The `sandbox::viewer` target adds the sample frame loop and reference scene. Public headers document who owns resources and when they may be replaced.

Voxel storage, chunk meshing, world generation, collision, streaming policy and saves belong in the game application. Use the sample's per-frame mesh replacement as a reference for safe chunk uploads. See [new-project guidance](docs/new-project.md).
