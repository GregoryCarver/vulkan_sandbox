add_library(sandbox_core STATIC
    src/core/device.cpp
    src/presentation/swapchain.cpp
    src/gpu/resources.cpp src/gpu/vma.cpp)
target_include_directories(sandbox_core PUBLIC include)
target_compile_features(sandbox_core PUBLIC cxx_std_20)
target_include_directories(sandbox_core SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/third_party/vma/include")
target_link_libraries(sandbox_core PUBLIC Vulkan::Vulkan)
if(ANDROID)
    target_link_libraries(sandbox_core PRIVATE log)
endif()
if(MSVC)
    target_compile_options(sandbox_core PRIVATE /W4 /permissive- /utf-8)
    target_compile_definitions(sandbox_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    target_compile_options(sandbox_core PRIVATE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
endif()


add_library(sandbox::core ALIAS sandbox_core)
