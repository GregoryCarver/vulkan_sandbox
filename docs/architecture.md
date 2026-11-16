# Architecture

The template shares Vulkan setup and GPU resource management between a Windows application and an Android application. Platform code owns the native window and operating-system integration; the sample owns camera behavior, cube geometry, materials, and debug UI. Applications can use the core without building the sample.

```text
engine/
  CMakeLists.txt                 Reusable sandbox::core target
  include/sandbox/
    core/                       Platform contract, Device, selection policies
    gpu/                        Allocator, Buffer, Image, Sampler, upload batches
    presentation/               Swapchain and presentation-resource retirement
  src/                          Implementations matching the public modules
samples/viewer/
  CMakeLists.txt                 Optional sandbox::viewer target and ImGui dependency
  include/sandbox/viewer/        Viewer, Scene, input, persistent view state
  src/                          Frame orchestration and the reference 3D scene
platforms/
  windows/                      GLFW adapter, desktop entry point, executable target
  android/                      GameActivity adapter and Android build/package project
assets/shaders/                 Sample Slang source; generated SPIR-V stays in the build tree
cmake/                          Host shader-compilation helper
tests/                          CPU regression tests
docs/                           Architecture, platform setup, and reuse guidance
third_party/                    Dependencies pinned by Git submodule commits
```

**Target boundaries are the module boundaries.** The root CMake file chooses which targets to build. `engine/CMakeLists.txt` defines `sandbox::core`; `samples/viewer/CMakeLists.txt` defines the optional sample and its UI dependency; each platform directory owns its application target. `SANDBOX_BUILD_SAMPLES=OFF` removes sample, GLFW, ImGui, and shader-compiler requirements. VMA and the Vulkan headers/loader remain core dependencies.

| Component | Owns | Does not own |
| --- | --- | --- |
| `Device` | Vulkan instance, debug messenger, surface, logical device, selected queues | Native window or application loop |
| `Allocator` and resource owners | GPU allocations, buffers, images/views, samplers | Scene objects or resource-use scheduling |
| `UploadQueue` | Upload command pool, pending submissions, staging allocations, completion fences | Destination resource lifetime |
| `Swapchain` | Swapchain images/views and presentation synchronization | Frame-slot command buffers, scene depth targets, camera |
| `Viewer` | Frame slots, scene/depth targets, resize handling, submission/presentation sequence | Native events, game world, application persistence |
| `Scene` | Reference geometry, materials, camera update, per-frame data, optional ImGui | Native-window lifecycle |
| Platform adapter | Native window, events, assets, writable paths, application startup | Shared rendering decisions |

The core uses the Vulkan C API directly, with narrow C++ RAII owners around resources. Vulkan handles, structures, command buffers, and synchronization remain available to callers. The previous Vulkan-Hpp tutorial implementation has been replaced by these owners; there is no additional graphics-API abstraction or hidden render graph. Ownership and lifetime policies are explicit in the public headers.

**Destruction follows dependency order.** The native window remains valid until the `Device` surface has been destroyed. `Device` outlives `Allocator`, `UploadQueue`, `Swapchain`, and all GPU resources. `Allocator` outlives its buffers and images. The upload queue outlives batches recorded through it. `Buffer`, `Image`, and `Sampler` are move-only; destroying them does not wait for GPU work.

`Viewer` stops GPU work before releasing its scene and frame resources. Swapchain destruction then retires presentation resources, followed by upload bookkeeping, allocator, device, surface, and instance. The explicit `Viewer::shutdown()` path returns the validation-error count after teardown; it ends the viewer's usable lifetime. Destructor cleanup also handles exceptional exits.

**Frame resources and swapchain images have different lifetimes.** The sample uses two frame slots. Each slot owns a command pool/buffer, completion fence, acquisition semaphore, and depth image. Render-finished semaphores belong to acquired swapchain images. An image index must never be substituted for a frame-slot index.

A frame waits for its slot's fence, collects completed uploads, acquires an image, updates only that slot's resources, records commands, submits, and presents. The slot fence is reset only when a submission is about to use it. A zero-sized surface or out-of-date swapchain skips presentation and leaves recreation for a subsequent frame.

Recreation waits for rendering to finish, rebuilds the swapchain and size-dependent attachments, and informs the scene about format/image-count changes. The ImGui context and CPU view state survive recreation. At reduced render scale, the scene renders to an offscreen color target and is blitted to the native-sized image; UI renders afterward at native resolution. When the surface format cannot support that blit path, the viewer falls back to native scene resolution.

**Presentation completion is tracked separately from rendering completion.** When supported, the device enables `VK_EXT_swapchain_maintenance1` and its instance dependencies. Each swapchain image receives a present fence; a reused fence is waited/reset, and pending present fences are waited before a generation's resources are destroyed. Enqueued out-of-date and surface-lost presentations are tracked too, while allocation failures are not treated as submitted fence operations.

Without that extension, retired generations are retained until an acquisition fence proves a previously presented image from the new generation has returned. This follows the [Khronos swapchain-recreation approach](https://docs.vulkan.org/samples/latest/samples/api/swapchain_recreation/README.html). Shutdown on the legacy path uses device-idle cleanup; it does not claim the explicit presentation-completion proof provided by maintenance1 fences.

**Uploads are batched and run on the rendering thread.** `UploadQueue` submits to the same graphics queue used by the viewer. Callers serialize queue access; this API is not a background-worker submission system. Workers may prepare CPU mesh data, but the rendering thread records and submits the GPU upload.

```cpp
auto batch = uploads.begin();
batch.upload(vertex_buffer, vertex_bytes);
batch.upload(index_buffer, index_bytes);
auto ticket = uploads.submit(std::move(batch));

// Poll in subsequent frames; reserve wait(ticket) for startup or explicit waits.
if (uploads.is_complete(ticket)) {
    // The upload has completed. Rendering may still own the old resource.
}
```

Each batch retains its staging buffers until its fence completes. Transfer barriers make the new contents available to subsequent graphics work. Mapped host writes use VMA flush/invalidate operations, including on noncoherent memory. `upload_rgba8` is an initial upload for one-layer, one-mip RGBA8 textures; applications needing mip chains, compression, arrays, or compute consumption must provide the corresponding copy/barrier policy.

The destination remains the caller's responsibility throughout an upload. Do not destroy it, or overwrite a resource still used by an earlier draw. In the sample, every frame slot has an active mesh and an optional replacement. After the slot's draw fence has completed, `Scene::prepare()` installs a replacement only when its upload ticket has also completed. That provides a concrete pattern for replacing chunk meshes without queue-wide idle waits on each copy. Startup uploads may wait for their ticket once.

**Assets and writable files use different interfaces.** `Platform::read_asset()` returns immutable bytes from a logical relative path. Windows resolves those paths beneath the application's asset directory; Android reads packaged assets. Runtime shader compilation is not required: Slang runs on the build host, and the compiled SPIR-V is copied into the executable package or APK.

`Platform::writable_path()` identifies application-owned storage for settings such as ImGui state; it may return an empty path when storage is unavailable. Callers must not attempt to save into the installed asset directory or assume APK assets are ordinary filesystem files. The platform also supplies framebuffer extent and UI scale.

**Input and lifecycle stay at the platform boundary.** Windows starts through a GLFW window and event loop. Android starts through GameActivity and supplies an `ANativeWindow` when its surface is available. The Android adapter suspends drawing when the activity cannot render and releases/recreates surface-dependent GPU state when the native window changes. CPU `ViewState` is kept outside the viewer so camera/UI choices can survive that GPU reconstruction.

Shared input describes pointer/key transitions, text, movement axes, and look deltas. The scene applies these actions after ImGui capture decisions. Touch gestures are mapped by the Android adapter; camera logic does not inspect Android events. Button/key events are retained separately from current states so a press and release between rendered frames can still reach the UI.

**The current device tier is deliberately explicit.** Runtime selection requires a Vulkan 1.3 device with dynamic rendering, synchronization2, graphics, and presentation support. Android OS version alone is not proof of these capabilities. The sample currently requires identity surface transform so rendering, UI, and input share upright coordinates; a future pre-rotation path must transform all three consistently. These are compatibility limits, not platform-test results.

Windows and Android are the implemented application adapters. Future Linux, macOS, or other adapters should implement native lifecycle, surface creation, assets/storage, input, and packaging while reusing the shared core. Vulkan-on-Metal integration, platform-specific feature restrictions, and real hardware validation remain work for those ports. The current `Device` constructor also requires a presentation surface; a headless/compute-only device policy is not exposed yet.

See [starting another project](new-project.md) for target-level reuse and the root [README](../README.md) for build and verification instructions.
