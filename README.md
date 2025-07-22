# Vulkan Sandbox

A minimal Vulkan project sandbox using C++23, GLFW 3.4, GLM 1.0.1, and IMGUI v1.92.1 — with shader compilation using [Slang](https://github.com/shader-slang/slang).  
All dependencies are self-contained and managed via Git submodules.

---

## 🛠️ Getting Started

### ✅ Clone the repository with submodules:
```bash
git clone --recurse-submodules https://github.com/your-username/vulkan_sandbox.git
```

Or, if you already cloned it:
```bash
git submodule update --init --recursive
```

---

## 🧱 Build Instructions

**Windows (PowerShell)**
```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

**Linux / macOS**
```bash
mkdir build
cd build
cmake ..
make
```

---

## 🔦 Shader Compilation

This project uses the `slangc` compiler from the Vulkan SDK. The shader is located at:
```
shaders/shader.slang
```
It is automatically compiled to:
```
shaders/compiled_shaders/slang.spv
```
on build.

Make sure the `VULKAN_SDK` environment variable is set correctly so `slangc` can be found.

---

## 📁 Project Structure

```
vulkan_sandbox/
├── shaders/
│   ├── shader.slang
│   └── compiled_shaders/
├── src/
│   └── main.cpp
├── third_party/
│   ├── glfw/   ← Git submodule
│   └── glm/    ← Git submodule
│   └── imgui/  ← Git submodule
├── CMakeLists.txt
└── README.md
```

---

## 🧩 Dependencies

- [GLFW](https://github.com/glfw/glfw) (submodule)
- [GLM](https://github.com/g-truc/glm) (submodule)
- [ImGui](https://github.com/ocornut/imgui) (submodule)
- [Slang](https://github.com/shader-slang/slang) (via Vulkan SDK)

---

## 💡 Notes

- You can update the submodules later with:
  ```bash
  git submodule update --remote
  ```

- Be sure to configure your IDE to use the Vulkan SDK path or run CMake from a terminal where it's already set.
