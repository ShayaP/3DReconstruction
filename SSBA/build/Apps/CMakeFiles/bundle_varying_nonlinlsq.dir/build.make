# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/build

# Include any dependencies generated for this target.
include Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/depend.make

# Include the progress variables for this target.
include Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/progress.make

# Include the compile flags for this target's objects.
include Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/flags.make

Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.o: Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/flags.make
Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.o: ../Apps/bundle_varying_nonlinlsq.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.o"
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/build/Apps && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.o -c /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/Apps/bundle_varying_nonlinlsq.cpp

Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.i"
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/build/Apps && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/Apps/bundle_varying_nonlinlsq.cpp > CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.i

Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.s"
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/build/Apps && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/Apps/bundle_varying_nonlinlsq.cpp -o CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.s

Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.o.requires:

.PHONY : Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.o.requires

Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.o.provides: Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.o.requires
	$(MAKE) -f Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/build.make Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.o.provides.build
.PHONY : Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.o.provides

Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.o.provides.build: Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.o


# Object files for target bundle_varying_nonlinlsq
bundle_varying_nonlinlsq_OBJECTS = \
"CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.o"

# External object files for target bundle_varying_nonlinlsq
bundle_varying_nonlinlsq_EXTERNAL_OBJECTS =

Apps/bundle_varying_nonlinlsq: Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.o
Apps/bundle_varying_nonlinlsq: Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/build.make
Apps/bundle_varying_nonlinlsq: libV3D.a
Apps/bundle_varying_nonlinlsq: libcolamd.a
Apps/bundle_varying_nonlinlsq: Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bundle_varying_nonlinlsq"
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/build/Apps && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bundle_varying_nonlinlsq.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/build: Apps/bundle_varying_nonlinlsq

.PHONY : Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/build

Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/requires: Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/bundle_varying_nonlinlsq.cpp.o.requires

.PHONY : Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/requires

Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/clean:
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/build/Apps && $(CMAKE_COMMAND) -P CMakeFiles/bundle_varying_nonlinlsq.dir/cmake_clean.cmake
.PHONY : Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/clean

Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/depend:
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/Apps /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/build /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/build/Apps /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/build/Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Apps/CMakeFiles/bundle_varying_nonlinlsq.dir/depend

