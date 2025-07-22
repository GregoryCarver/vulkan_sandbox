cmake_minimum_required(VERSION 4.0)
project(vulkan_sandbox LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

set(SLANGC_EXECUTABLE $ENV{VULKAN_SDK}//bin/slangc)
set(SHADER_SLANG_SOURCES ${CMAKE_CURRENT_LIST_DIR}/shaders/shader.slang)

function(add_slang_shader_target TARGET)
    cmake_parse_arguments(SHADER "" "" "SOURCES" ${ARGN})
    set(SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders/compiled_shaders/)
    set(ENTRY_POINTS -entry vertMain -entry fragMain)
    add_custom_command(
        OUTPUT  ${SHADERS_DIR}/slang.spv
        COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES}
                -target spirv -profile spirv_1_4 -emit-spirv-directly
                -fvk-use-entrypoint-name ${ENTRY_POINTS}
                -o ${SHADERS_DIR}/slang.spv
        WORKING_DIRECTORY ${SHADERS_DIR}
        DEPENDS ${SHADER_SOURCES}
        COMMENT "Compiling Slang Shaders"
        VERBATIM
    )
    add_custom_target(${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()

# GLFW and GLM are added as submodules under third_party/
add_subdirectory(third_party/glfw)
add_subdirectory(third_party/glm)

add_executable(${PROJECT_NAME}
  src/main.cpp
)

add_slang_shader_target(shader_spv SOURCES ${SHADER_SLANG_SOURCES})
add_dependencies(${PROJECT_NAME} shader_spv)

target_compile_definitions(${PROJECT_NAME}
  PRIVATE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
)

target_include_directories(${PROJECT_NAME}
  PRIVATE ${Vulkan_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE glfw glm Vulkan::Vulkan
)

if (WIN32)
  target_compile_definitions(${PROJECT_NAME} PRIVATE VK_USE_PLATFORM_WIN32_KHR)
elseif (APPLE)
  target_compile_definitions(${PROJECT_NAME} PRIVATE VK_USE_PLATFORM_MACOS_MVK)
elseif (UNIX)
  target_compile_definitions(${PROJECT_NAME} PRIVATE VK_USE_PLATFORM_XCB_KHR)
endif()
