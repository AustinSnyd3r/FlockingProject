# How to Build and Run CudaOpenGLFlocking

Open PowerShell in the project root (C:\Users\{user}\CLionProjects\CudaOpenGLFlocking). If you want a clean build, delete the build folder by running:

```powershell
rm -r -fo build
```

Then configure the project with CMake by running:

```powershell
cmake -B build -S .
```
 
Then build it:

```powershell
cmake --build build
```

If the build finishes successfully, the executable will be located at:

```
build\Debug\CudaOpenGLFlocking.exe
```

To run it:

```powershell
.\build\Debug\CudaOpenGLFlocking.exe
```

If you get an error saying that `glew32d.dll` is missing, copy it manually from:

```
build\bin\Debug\glew32d.dll
```

into:

```
build\Debug\
```

If you ever see a fatal error about missing `glfw3.lib` or `glew_static.lib`, it means you did not add the subdirectories correctly in your CMakeLists.txt. Make sure you have:

```cmake
add_subdirectory(libs/glew/build/cmake)
add_subdirectory(libs/glfw)
```

Also make sure you are linking to the correct targets:

```cmake
target_link_libraries(${PROJECT_NAME} glew_s glfw opengl32)
```

If you need to completely rebuild and run quickly, the full set of commands is:

```powershell
rm -r -fo build
cmake -B build -S .
cmake --build build
.\build\Debug\CudaOpenGLFlocking.exe
```

GLEW, GLFW, and OpenGL are all linked automatically now. CUDA separable compilation is enabled for your CUDA sources. Debug builds use dynamic runtime (/MDd). Ignore linker warnings like LIBCMT conflicts during Debug builds. End of instructions.