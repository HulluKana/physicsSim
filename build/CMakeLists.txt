cmake_minimum_required(VERSION "3.22.1")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_custom_target(shaders)

file(GLOB_RECURSE SHADERS_SRC "../Shaders/*.vert" "../Shaders/*.frag" "../Shaders/*.comp")
foreach(FILE ${SHADERS_SRC})
    get_filename_component(FILE_NAME ${FILE} NAME)
    set(COMPILED_FILE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${FILE_NAME}.spv")
    if (${FILE} IS_NEWER_THAN ${COMPILED_FILE})
        add_custom_command( TARGET shaders
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                            DEPENDS ${FILE}
                            COMMAND glslc --target-env=vulkan1.3 ${FILE} -o ${COMPILED_FILE}
                            COMMENT "Compiling ${FILE_NAME}"
        )
    endif()
endforeach(FILE)



project(physicsSim)

file(GLOB_RECURSE VUL_SRC "../vulkano/essentials/src/*.cpp" "../vulkano/optionals/src/*.cpp")
file(GLOB_RECURSE SRC "../src/*.cpp")
add_executable(physicsSim "${SRC}" "${VUL_SRC}")
add_dependencies(physicsSim shaders)
target_compile_options(physicsSim PRIVATE "-Wall" "-g")

add_compile_definitions(VUL_ENABLE_DEBUG_NAMER)

target_link_libraries(physicsSim vulkan)
target_link_libraries(physicsSim glfw)
target_link_libraries(physicsSim ktx)
target_link_libraries(physicsSim OpenEXR-3_2)

set(IMGUI_PATH "../vulkano/3rdParty/imgui")
file(GLOB IMGUI_SOURCES ${IMGUI_PATH}/*.cpp) 
add_library("ImGui" STATIC ${IMGUI_SOURCES})
target_include_directories("ImGui" PUBLIC ${IMGUI_PATH})
target_link_libraries(physicsSim ImGui)

target_include_directories(physicsSim PUBLIC "../include" "../vulkano/3rdParty/"  "../vulkano/essentials/include/" "../vulkano/optionals/include/")
