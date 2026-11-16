# Validation record

Recorded on 2026-11-16 for `codex/windows-android-template`, based on the latest
`origin/master` commit `806c93a`. This records checks actually performed; platform
source code and CI configuration are not substitutes for successful builds or
device runs.

## Windows results

Environment: Windows x64, Visual Studio 2022 / MSVC 19.44, CMake 4.0.3, Vulkan SDK
1.4.313.2 (including Slang), NVIDIA GeForce RTX 3060 Laptop GPU. The driver reported
Vulkan 1.4.351. The application requests Vulkan 1.3.

| Check | Result |
| --- | --- |
| Debug build | Passed |
| Release build | Passed |
| Release with `SANDBOX_ENABLE_IMGUI=OFF` | Passed |
| Core-only Release with `SANDBOX_BUILD_SAMPLES=OFF` | Passed; no sample/UI/shader targets |
| Debug and Release CPU tests | 2/2 passed in each: device/surface selection and ImGui input transitions |
| No-ImGui and core-only CPU tests | 1/1 passed in each: device/surface selection |
| Debug GPU smoke, 240 frames, streaming meshes, render scale 0.75 | Passed; validation enabled, zero errors including teardown |
| Installed Release package, 240 frames, legacy presentation, streaming meshes | Passed; validation enabled, zero errors including teardown |
| ImGui-free Release runtime, 60 frames, streaming meshes | Passed; validation enabled, zero errors including teardown |
| Synchronization validation: 240-frame scaled and 240-frame legacy smoke runs | Both passed; zero errors including teardown |
| Interactive visual check | Textured cubes, depth, camera and ImGui rendered; clean close, zero logged errors |
| SPIR-V validation for Vulkan 1.3 | Passed |
| Clean local checkout of implementation commit `21e7689` | Debug and Release built; 2/2 CPU tests passed in each; generated SPIR-V validated |

The clean checkout at `out/clean-checkout` cloned the committed source and its exact
submodule revisions from existing local repositories. It used new build directories
and no previous generated assets. This verifies source/dependency completeness;
it does not test downloading dependencies from GitHub on a new machine.

The smoke sequence resizes through 960x540, 1024x768 and 1280x800, minimizes and
restores the window, continuously replaces mesh buffers, and checks shutdown.
Runs started with `C:/Dev` as the working directory, outside the repository.
The packaged run used `out/package/windows/sandbox_windows.exe` and its copied
assets. The reduced-resolution run kept UI at native framebuffer resolution.

Local logs are ignored build artifacts under `out/windows/`: `smoke-debug-final.log`,
`smoke-release-legacy.log`, `smoke-no-ui.log`, `smoke-sync-scaled.log`,
`smoke-sync-legacy.log`, `visual.log` and `visual-errors.log`.
The interactive visual check preceded the final input/diagnostic refinements;
the automated runs cover the updated renderer.

Reproduce the normal GPU check using `tools/verify-windows.ps1`. Pass
`-Configuration Release`, `-RenderScale 0.75`, or `-LegacyPresent` as needed.
Synchronization validation can be enabled before launching with
`$env:VK_LAYER_ENABLES = 'VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT'`.
A machine without the Khronos validation layer can render the sample, but that
does not constitute a validation-enabled run.

## Android and CI status

The GameActivity adapter, Gradle configuration, packaged-asset pipeline and
ARM64 target are implemented. XML parsing and the missing-toolchain launcher
diagnostic were checked. Native/Gradle API usage was reviewed against Android's
official GameActivity documentation.

**Android native compilation, APK assembly, installation and runtime testing
remain unverified.** The workspace had no Android SDK/NDK, JDK 17 or Gradle; the
toolchain download was canceled. The wrapper JAR also remains to be generated.
The Galaxy S24 is the selected first device, but no physical-device result is
claimed. Follow the [Android setup and device checklist](android.md).

The GitHub Actions workflow builds Windows configurations and an Android debug
APK. It is added locally and has not run on GitHub. Hosted build success would
still leave physical GPU/lifecycle checks outstanding.

## Limits of the evidence

- Desktop tests cover one NVIDIA GPU/driver, not every Windows GPU or separate
  graphics/present queue configuration.
- Legacy presentation teardown uses device idle; explicit presentation-completion
  fences are used only when swapchain maintenance is available.
- Android currently requires API 33+, Vulkan 1.3 and identity surface transform.
  Touch, soft-keyboard behavior, rotation, pause/resume and thermal/frame-pacing
  behavior need verification on the Galaxy S24.
- Linux and macOS application adapters are future work.
