# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kohigashi/projects/learning_ocp/mpc_1dof

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kohigashi/projects/learning_ocp/mpc_1dof/build

# Include any dependencies generated for this target.
include CMakeFiles/mpc_1dof.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/mpc_1dof.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/mpc_1dof.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mpc_1dof.dir/flags.make

CMakeFiles/mpc_1dof.dir/codegen:
.PHONY : CMakeFiles/mpc_1dof.dir/codegen

CMakeFiles/mpc_1dof.dir/mpc.cpp.o: CMakeFiles/mpc_1dof.dir/flags.make
CMakeFiles/mpc_1dof.dir/mpc.cpp.o: /home/kohigashi/projects/learning_ocp/mpc_1dof/mpc.cpp
CMakeFiles/mpc_1dof.dir/mpc.cpp.o: CMakeFiles/mpc_1dof.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/kohigashi/projects/learning_ocp/mpc_1dof/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mpc_1dof.dir/mpc.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mpc_1dof.dir/mpc.cpp.o -MF CMakeFiles/mpc_1dof.dir/mpc.cpp.o.d -o CMakeFiles/mpc_1dof.dir/mpc.cpp.o -c /home/kohigashi/projects/learning_ocp/mpc_1dof/mpc.cpp

CMakeFiles/mpc_1dof.dir/mpc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/mpc_1dof.dir/mpc.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kohigashi/projects/learning_ocp/mpc_1dof/mpc.cpp > CMakeFiles/mpc_1dof.dir/mpc.cpp.i

CMakeFiles/mpc_1dof.dir/mpc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/mpc_1dof.dir/mpc.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kohigashi/projects/learning_ocp/mpc_1dof/mpc.cpp -o CMakeFiles/mpc_1dof.dir/mpc.cpp.s

# Object files for target mpc_1dof
mpc_1dof_OBJECTS = \
"CMakeFiles/mpc_1dof.dir/mpc.cpp.o"

# External object files for target mpc_1dof
mpc_1dof_EXTERNAL_OBJECTS =

mpc_1dof: CMakeFiles/mpc_1dof.dir/mpc.cpp.o
mpc_1dof: CMakeFiles/mpc_1dof.dir/build.make
mpc_1dof: CMakeFiles/mpc_1dof.dir/compiler_depend.ts
mpc_1dof: /usr/local/lib/libOsqpEigen.so.0.8.1
mpc_1dof: /usr/lib/libosqp.so
mpc_1dof: CMakeFiles/mpc_1dof.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/kohigashi/projects/learning_ocp/mpc_1dof/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mpc_1dof"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mpc_1dof.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mpc_1dof.dir/build: mpc_1dof
.PHONY : CMakeFiles/mpc_1dof.dir/build

CMakeFiles/mpc_1dof.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mpc_1dof.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mpc_1dof.dir/clean

CMakeFiles/mpc_1dof.dir/depend:
	cd /home/kohigashi/projects/learning_ocp/mpc_1dof/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kohigashi/projects/learning_ocp/mpc_1dof /home/kohigashi/projects/learning_ocp/mpc_1dof /home/kohigashi/projects/learning_ocp/mpc_1dof/build /home/kohigashi/projects/learning_ocp/mpc_1dof/build /home/kohigashi/projects/learning_ocp/mpc_1dof/build/CMakeFiles/mpc_1dof.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/mpc_1dof.dir/depend

