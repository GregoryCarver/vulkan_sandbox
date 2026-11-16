set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/glfw" "${PROJECT_BINARY_DIR}/third_party/glfw" EXCLUDE_FROM_ALL)
add_executable(sandbox_windows main.cpp windows_platform.cpp)
target_link_libraries(sandbox_windows PRIVATE sandbox::viewer glfw)
set_target_properties(sandbox_windows PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:sandbox_windows>")
# This target runs on every build, including shader-only edits where the exe
# does not relink. Runtime assets therefore cannot silently become stale.
add_custom_target(sandbox_windows_assets ALL
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${SANDBOX_ASSET_DIRECTORY}" "$<TARGET_FILE_DIR:sandbox_windows>/assets"
    DEPENDS sandbox_shaders)
add_dependencies(sandbox_windows sandbox_windows_assets)
install(TARGETS sandbox_windows RUNTIME DESTINATION .)
install(DIRECTORY "${SANDBOX_ASSET_DIRECTORY}/" DESTINATION assets)
