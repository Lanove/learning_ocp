# Specify the minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(ConvexMPC)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PROJECT_VERSION 1.0.0)

# Find dependencies
find_package(OsqpEigen REQUIRED)
find_package(Eigen3 REQUIRED)

# Define the header-only library
add_library(ConvexMPC INTERFACE)

# Link OsqpEigen privately
target_link_libraries(ConvexMPC INTERFACE OsqpEigen::OsqpEigen)

# Set include directories for clients of the library
target_include_directories(ConvexMPC INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> 
    $<INSTALL_INTERFACE:include/ConvexMPC>  # Path after installation
)

# Install headers to /usr/local/include/ConvexMPC
install(DIRECTORY include/ DESTINATION include/ConvexMPC)

# Generate and install the CMake config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ConvexMPCConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Create a custom ConvexMPCConfig.cmake for dependencies
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ConvexMPCConfig.cmake" "
include(CMakeFindDependencyMacro)
find_dependency(OsqpEigen REQUIRED)
include(\"\${CMAKE_CURRENT_LIST_DIR}/ConvexMPCTargets.cmake\")
")

# Install the generated config and version files
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ConvexMPCConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/ConvexMPCConfigVersion.cmake"
        DESTINATION lib/cmake/ConvexMPC)
