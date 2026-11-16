# slangc runs on the build HOST, even when CMake targets Android.
find_program(SLANGC_EXECUTABLE NAMES slangc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    NO_CMAKE_FIND_ROOT_PATH REQUIRED)
set(SANDBOX_ASSET_DIRECTORY "${CMAKE_BINARY_DIR}/assets" CACHE PATH "Generated runtime assets")
function(sandbox_compile_shader target source relative_output)
    cmake_parse_arguments(SHADER "" "PROFILE" "ENTRY_POINTS;DEPENDS" ${ARGN})
    if(NOT SHADER_PROFILE)
        set(SHADER_PROFILE spirv_1_5)
    endif()
    if(NOT SHADER_ENTRY_POINTS)
        set(SHADER_ENTRY_POINTS vertMain:vertex fragMain:fragment)
    endif()
    set(entry_arguments)
    foreach(entry IN LISTS SHADER_ENTRY_POINTS)
        if(NOT entry MATCHES "^([^:]+):([^:]+)$")
            message(FATAL_ERROR "Shader entry must be name:stage, got '${entry}'")
        endif()
        list(APPEND entry_arguments -entry "${CMAKE_MATCH_1}" -stage "${CMAKE_MATCH_2}")
    endforeach()
    set(output "${SANDBOX_ASSET_DIRECTORY}/${relative_output}")
    get_filename_component(output_directory "${output}" DIRECTORY)
    get_filename_component(source_directory "${source}" DIRECTORY)
    file(GLOB_RECURSE dependencies CONFIGURE_DEPENDS "${source_directory}/*.slang")
    add_custom_command(OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_directory}"
        COMMAND "${SLANGC_EXECUTABLE}" "${source}"
            -target spirv -profile "${SHADER_PROFILE}" -emit-spirv-directly
            -matrix-layout-column-major -fvk-use-entrypoint-name
            ${entry_arguments}
            -o "${output}"
        DEPENDS "${source}" ${dependencies} ${SHADER_DEPENDS}
        COMMENT "Compiling ${relative_output}"
        VERBATIM)
    add_custom_target(${target} ALL DEPENDS "${output}")
endfunction()
