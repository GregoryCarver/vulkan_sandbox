# Android application

The Android adapter builds the same `sandbox_viewer` sample as Windows. Android owns
the Activity and native window; the shared core owns Vulkan objects. All Android/JNI
code stays in `platforms/android`, outside the engine and sample rendering code.

## Initial device tier

- ARM64 Android 13 / API 33 or newer. This matches the minimum OS that exports
  the directly linked Vulkan 1.3 loader functions; see the
  [Android Vulkan version requirements](https://developer.android.com/games/develop/vulkan/native-engine-support).
- Vulkan 1.3, dynamic rendering, synchronization2, and presentation support are
  required at runtime. Android version alone does not establish Vulkan support.
- The manifest declares Vulkan 1.3 so distribution can filter incompatible devices.
- The user's Galaxy S24 is the initial physical test target. Its GPU/driver must be
  queried on the actual device; this repository does not infer Snapdragon/Exynos
  or claim physical validation from a successful desktop build.

## Pinned build tools

| Component | Version |
| --- | --- |
| JDK | 17 |
| Gradle | 8.11.1 |
| Android Gradle Plugin | 8.9.2 |
| Android compile / target SDK | 35 |
| Android NDK | 28.0.13004108 |
| CMake for Android | 3.31.6 |
| AndroidX GameActivity | 3.0.5 |

Install these with Android Studio's SDK Manager or `sdkmanager`. NDK r28 supplies
16 KB page-compatible native builds by default. Use the GameActivity-provided
native app glue through Prefab; the NDK's similarly named `native_app_glue` has
a different input API and is not interchangeable.

Set `JAVA_HOME` to JDK 17 and `ANDROID_HOME` to your SDK. Alternatively, set `sdk.dir`
in the ignored `platforms/android/local.properties` file. Install `platform-tools`
for `adb` and the Windows USB driver appropriate to the connected phone.

The initial workspace had no Android SDK/NDK or JDK 17. The dependency download
was canceled, so the official Gradle wrapper JAR has not been retrieved. The
included launcher can use an installed Gradle. To generate the official wrapper:

```powershell
cd platforms/android
gradle wrapper --gradle-version 8.11.1 --distribution-type bin
```

Commit the generated wrapper JAR when bootstrapping the project's Android toolchain;
the launcher/properties alone are not a fully self-contained wrapper.

## Build, install, and run

Initialize repository submodules first. The Slang compiler runs on the host PC,
even while CMake cross-compiles C++ for Android. It is found using `VULKAN_SDK`,
the host PATH, or the `slangc` Gradle property.

```powershell
cd platforms/android
.\gradlew.bat :app:assembleDebug
# An explicit Slang compiler can be supplied if needed:
.\gradlew.bat :app:assembleDebug '-Pslangc=C:/VulkanSDK/1.4.313.2/Bin/slangc.exe'

adb devices -l
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n dev.vulkansandbox.viewer/.MainActivity
adb logcat -s VulkanSandbox
```

USB debugging must be enabled and the device must authorize the development PC.
An `unauthorized` device is not usable until the phone's confirmation is accepted.

`sandbox_shaders` compiles the sample's SPIR-V through the same CMake function used
on Windows. `sandbox_android_assets` copies it into a variant-specific generated
asset directory. Gradle's asset-merge task waits for the native build, so a shader
edit is packaged even when C++ sources do not change. Shader assets are read with
`AAssetManager`; settings are written under the app's private files directory.

The sample uses FIFO presentation with a 60 FPS application cap. Change the cap
and internal resolution without changing the shared renderer:

```powershell
.\gradlew.bat :app:assembleDebug -PsandboxFps=30 -PsandboxRenderScale=0.75
```

This is an application frame cap, not Android Frame Pacing / Swappy integration.
Measure frame delivery and thermal behavior before introducing a dedicated mobile
frame-pacing implementation.

## Input and application lifecycle

- The bottom 40% of the surface is reserved for camera gestures: drag on the left
  to move, and drag on the right to look. The upper region accepts ImGui pointer
  interaction. Pointer positions are framebuffer pixels; movement sensitivity
  and UI scaling account for Android density.
- Physical keyboard controls are WASD to move, Q/E down/up, and F1 to toggle UI.
  A mouse supports UI clicks and right-button camera look.
- GameActivity input is translated into shared `InputState`, including ordered
  key/button transitions so a quick tap between rendered frames is retained.
  Each event is submitted once; the Viewer retains events if a resize delays drawing.
- When an ImGui text field requests text input, the adapter shows Android's soft
  keyboard and forwards edits within that keyboard session. The minimal text
  bridge is intended for simple debug fields: it does not synchronize an existing
  field's contents, cursor/selection changes, or switching fields while the IME
  stays open. Supplementary Unicode/composition input also needs device testing.
  A game chat or naming UI should integrate its text model with GameTextInput
  directly rather than use this debug-input bridge.
- The native loop blocks in `ALooper_pollOnce` when paused, unfocused, or without a
  valid surface. It does not busy-spin in the background.
- `APP_CMD_TERM_WINDOW` shuts down the Viewer before native glue acknowledges
  window destruction. Camera/UI state is held outside the disposable GPU session
  and survives surface recreation. A versioned CPU state record supports Activity
  state restoration; GPU handles are never serialized.
- Renderer initialization errors are logged and shown on Android's UI thread.

## Validation status and physical-device checklist

Android native compilation, APK assembly, installation, and runtime behavior are
**not yet verified** in the initial implementation environment. The Android SDK,
NDK, GameActivity AAR, and Gradle dependency resolution must be available before
those checks can run. Windows tests do not establish Android correctness.

On a connected Galaxy S24, record the actual GPU/driver and Vulkan version, then:

1. Build/install the debug APK and confirm the textured, depth-tested scene and UI.
2. Exercise movement, look, short taps, keyboard input, and soft-keyboard text.
3. Rotate the device, background/foreground repeatedly, and lock/unlock the screen.
4. Close/relaunch and exercise Activity recreation; verify no stale Vulkan surface.
5. Enable mesh streaming, change resolution, and run long enough to inspect memory,
   frame delivery, battery/thermal behavior, and any driver errors.
6. Repeat with Vulkan validation layers explicitly installed/enabled on the phone;
   a missing validation layer must never be reported as a clean validation run.

Release signing, store publication, and production package identifiers belong to
the application built from this template, not to this debug sample.

## References

- [GameActivity integration](https://developer.android.com/games/agdk/game-activity/get-started)
- [GameActivity input and lifecycle](https://developer.android.com/games/agdk/game-activity/migrate-native-activity)
- [GameActivity text input](https://developer.android.com/games/agdk/game-activity/use-text-input)
- [Android Vulkan device requirements](https://developer.android.com/games/develop/vulkan/native-engine-support)
- [AGP 8.9 compatibility](https://developer.android.com/build/releases/agp-8-9-0-release-notes)
