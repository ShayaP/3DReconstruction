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
CMAKE_SOURCE_DIR = /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build

# Include any dependencies generated for this target.
include SSBA/Apps/CMakeFiles/bundle_common.dir/depend.make

# Include the progress variables for this target.
include SSBA/Apps/CMakeFiles/bundle_common.dir/progress.make

# Include the compile flags for this target's objects.
include SSBA/Apps/CMakeFiles/bundle_common.dir/flags.make

SSBA/Apps/CMakeFiles/bundle_common.dir/bundle_common.cpp.o: SSBA/Apps/CMakeFiles/bundle_common.dir/flags.make
SSBA/Apps/CMakeFiles/bundle_common.dir/bundle_common.cpp.o: ../SSBA/Apps/bundle_common.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object SSBA/Apps/CMakeFiles/bundle_common.dir/bundle_common.cpp.o"
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/SSBA/Apps && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bundle_common.dir/bundle_common.cpp.o -c /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/Apps/bundle_common.cpp

SSBA/Apps/CMakeFiles/bundle_common.dir/bundle_common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bundle_common.dir/bundle_common.cpp.i"
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/SSBA/Apps && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/Apps/bundle_common.cpp > CMakeFiles/bundle_common.dir/bundle_common.cpp.i

SSBA/Apps/CMakeFiles/bundle_common.dir/bundle_common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bundle_common.dir/bundle_common.cpp.s"
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/SSBA/Apps && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/Apps/bundle_common.cpp -o CMakeFiles/bundle_common.dir/bundle_common.cpp.s

SSBA/Apps/CMakeFiles/bundle_common.dir/bundle_common.cpp.o.requires:

.PHONY : SSBA/Apps/CMakeFiles/bundle_common.dir/bundle_common.cpp.o.requires

SSBA/Apps/CMakeFiles/bundle_common.dir/bundle_common.cpp.o.provides: SSBA/Apps/CMakeFiles/bundle_common.dir/bundle_common.cpp.o.requires
	$(MAKE) -f SSBA/Apps/CMakeFiles/bundle_common.dir/build.make SSBA/Apps/CMakeFiles/bundle_common.dir/bundle_common.cpp.o.provides.build
.PHONY : SSBA/Apps/CMakeFiles/bundle_common.dir/bundle_common.cpp.o.provides

SSBA/Apps/CMakeFiles/bundle_common.dir/bundle_common.cpp.o.provides.build: SSBA/Apps/CMakeFiles/bundle_common.dir/bundle_common.cpp.o


# Object files for target bundle_common
bundle_common_OBJECTS = \
"CMakeFiles/bundle_common.dir/bundle_common.cpp.o"

# External object files for target bundle_common
bundle_common_EXTERNAL_OBJECTS =

SSBA/Apps/bundle_common: SSBA/Apps/CMakeFiles/bundle_common.dir/bundle_common.cpp.o
SSBA/Apps/bundle_common: SSBA/Apps/CMakeFiles/bundle_common.dir/build.make
SSBA/Apps/bundle_common: SSBA/libV3D.a
SSBA/Apps/bundle_common: SSBA/libcolamd.a
SSBA/Apps/bundle_common: SSBA/Apps/CMakeFiles/bundle_common.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bundle_common"
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/SSBA/Apps && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bundle_common.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
SSBA/Apps/CMakeFiles/bundle_common.dir/build: SSBA/Apps/bundle_common

.PHONY : SSBA/Apps/CMakeFiles/bundle_common.dir/build

SSBA/Apps/CMakeFiles/bundle_common.dir/requires: SSBA/Apps/CMakeFiles/bundle_common.dir/bundle_common.cpp.o.requires

.PHONY : SSBA/Apps/CMakeFiles/bundle_common.dir/requires

SSBA/Apps/CMakeFiles/bundle_common.dir/clean:
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/SSBA/Apps && $(CMAKE_COMMAND) -P CMakeFiles/bundle_common.dir/cmake_clean.cmake
.PHONY : SSBA/Apps/CMakeFiles/bundle_common.dir/clean

SSBA/Apps/CMakeFiles/bundle_common.dir/depend:
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/SSBA/Apps /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/SSBA/Apps /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/SSBA/Apps/CMakeFiles/bundle_common.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : SSBA/Apps/CMakeFiles/bundle_common.dir/depend

