find_package(game-activity REQUIRED CONFIG)

add_library(sandbox_android SHARED
    native/android_main.cpp
    native/android_platform.cpp
    native/android_input.cpp)
target_include_directories(sandbox_android PRIVATE native)
target_compile_definitions(sandbox_android PRIVATE VK_USE_PLATFORM_ANDROID_KHR)
set(SANDBOX_ANDROID_TARGET_FPS "60" CACHE STRING "Android application frame-rate cap")
set(SANDBOX_ANDROID_RENDER_SCALE "1.0" CACHE STRING "Android internal render resolution scale")
if(NOT SANDBOX_ANDROID_TARGET_FPS MATCHES "^[0-9]+$" OR
   SANDBOX_ANDROID_TARGET_FPS LESS 15 OR SANDBOX_ANDROID_TARGET_FPS GREATER 240)
    message(FATAL_ERROR "SANDBOX_ANDROID_TARGET_FPS must be an integer from 15 to 240.")
endif()
if(NOT SANDBOX_ANDROID_RENDER_SCALE MATCHES "^[0-9]+(\\.[0-9]+)?$" OR
   SANDBOX_ANDROID_RENDER_SCALE LESS 0.25 OR SANDBOX_ANDROID_RENDER_SCALE GREATER 1.0)
    message(FATAL_ERROR "SANDBOX_ANDROID_RENDER_SCALE must be a number from 0.25 to 1.0.")
endif()
target_compile_definitions(sandbox_android PRIVATE
    SANDBOX_ANDROID_TARGET_FPS=${SANDBOX_ANDROID_TARGET_FPS}
    SANDBOX_ANDROID_RENDER_SCALE=${SANDBOX_ANDROID_RENDER_SCALE})
target_compile_features(sandbox_android PRIVATE cxx_std_20)
target_link_libraries(sandbox_android PRIVATE
    sandbox_viewer game-activity::game-activity_static android log)
# GameActivity discovers this native entrypoint through dlsym.
target_link_options(sandbox_android PRIVATE "-uGameActivity_onCreate")
target_compile_options(sandbox_android PRIVATE -Wall -Wextra -Wpedantic)

if(NOT SANDBOX_ANDROID_ASSET_OUTPUT)
    message(FATAL_ERROR "Configure Android through platforms/android Gradle project (missing asset output).")
endif()
add_custom_target(sandbox_android_assets
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${SANDBOX_ANDROID_ASSET_OUTPUT}"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${SANDBOX_ASSET_DIRECTORY}" "${SANDBOX_ANDROID_ASSET_OUTPUT}"
    DEPENDS sandbox_shaders)
add_dependencies(sandbox_android sandbox_android_assets)
