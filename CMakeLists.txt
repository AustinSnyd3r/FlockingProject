cmake_minimum_required(VERSION 3.10)
project(CudaOpenGLFlocking LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75)

# Add GLEW and GLFW as subdirectories
add_subdirectory(libs/glew/build/cmake)
add_subdirectory(libs/glfw)

# Define where to find headers
include_directories(
        ${CMAKE_SOURCE_DIR}/libs/glew/include
        ${CMAKE_SOURCE_DIR}/libs/glfw/include
        ${CMAKE_SOURCE_DIR}/libs/glm
        ${CMAKE_SOURCE_DIR}/src/include
)

# Define your sources
set(SOURCES
        main.cu
        rendering/RenderHelpers.cpp
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link against GLEW, GLFW, and OpenGL
target_link_libraries(${PROJECT_NAME}
        glew_s
        glfw
        opengl32
)

# Definitions needed for GLEW static linking
target_compile_definitions(${PROJECT_NAME} PRIVATE GLEW_STATIC)

# Enable CUDA separable compilation
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
