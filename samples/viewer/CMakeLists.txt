if(SANDBOX_ENABLE_IMGUI)
    add_library(sandbox_imgui STATIC
        ${PROJECT_SOURCE_DIR}/third_party/imgui/imgui.cpp ${PROJECT_SOURCE_DIR}/third_party/imgui/imgui_draw.cpp
        ${PROJECT_SOURCE_DIR}/third_party/imgui/imgui_tables.cpp ${PROJECT_SOURCE_DIR}/third_party/imgui/imgui_widgets.cpp
        ${PROJECT_SOURCE_DIR}/third_party/imgui/imgui_demo.cpp ${PROJECT_SOURCE_DIR}/third_party/imgui/backends/imgui_impl_vulkan.cpp)
    target_include_directories(sandbox_imgui SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/third_party/imgui ${PROJECT_SOURCE_DIR}/third_party/imgui/backends)
    target_link_libraries(sandbox_imgui PUBLIC Vulkan::Vulkan)
endif()

add_library(sandbox_viewer STATIC src/viewer.cpp src/scene.cpp)
target_include_directories(sandbox_viewer PUBLIC include)
target_compile_features(sandbox_viewer PUBLIC cxx_std_20)
target_link_libraries(sandbox_viewer PUBLIC sandbox_core PRIVATE glm::glm)
target_compile_definitions(sandbox_viewer PRIVATE GLM_FORCE_RADIANS GLM_FORCE_DEPTH_ZERO_TO_ONE)
target_compile_definitions(sandbox_viewer PRIVATE SANDBOX_ENABLE_IMGUI=$<BOOL:${SANDBOX_ENABLE_IMGUI}>)
if(SANDBOX_ENABLE_IMGUI)
    target_link_libraries(sandbox_viewer PRIVATE sandbox_imgui)
endif()
sandbox_compile_shader(sandbox_shaders "${PROJECT_SOURCE_DIR}/assets/shaders/cube.slang" "shaders/cube.spv")
add_dependencies(sandbox_viewer sandbox_shaders)


add_library(sandbox::viewer ALIAS sandbox_viewer)
