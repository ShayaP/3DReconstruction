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
include libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/depend.make

# Include the progress variables for this target.
include libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/progress.make

# Include the compile flags for this target's objects.
include libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/flags.make

libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/bundle_varying.cpp.o: libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/flags.make
libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/bundle_varying.cpp.o: ../libs/SSBA/Apps/bundle_varying.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/bundle_varying.cpp.o"
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/libs/SSBA/Apps && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bundle_varying.dir/bundle_varying.cpp.o -c /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/libs/SSBA/Apps/bundle_varying.cpp

libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/bundle_varying.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bundle_varying.dir/bundle_varying.cpp.i"
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/libs/SSBA/Apps && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/libs/SSBA/Apps/bundle_varying.cpp > CMakeFiles/bundle_varying.dir/bundle_varying.cpp.i

libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/bundle_varying.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bundle_varying.dir/bundle_varying.cpp.s"
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/libs/SSBA/Apps && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/libs/SSBA/Apps/bundle_varying.cpp -o CMakeFiles/bundle_varying.dir/bundle_varying.cpp.s

libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/bundle_varying.cpp.o.requires:

.PHONY : libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/bundle_varying.cpp.o.requires

libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/bundle_varying.cpp.o.provides: libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/bundle_varying.cpp.o.requires
	$(MAKE) -f libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/build.make libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/bundle_varying.cpp.o.provides.build
.PHONY : libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/bundle_varying.cpp.o.provides

libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/bundle_varying.cpp.o.provides.build: libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/bundle_varying.cpp.o


# Object files for target bundle_varying
bundle_varying_OBJECTS = \
"CMakeFiles/bundle_varying.dir/bundle_varying.cpp.o"

# External object files for target bundle_varying
bundle_varying_EXTERNAL_OBJECTS =

libs/SSBA/Apps/bundle_varying: libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/bundle_varying.cpp.o
libs/SSBA/Apps/bundle_varying: libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/build.make
libs/SSBA/Apps/bundle_varying: libs/SSBA/libV3D.a
libs/SSBA/Apps/bundle_varying: libs/SSBA/libcolamd.a
libs/SSBA/Apps/bundle_varying: libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bundle_varying"
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/libs/SSBA/Apps && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bundle_varying.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/build: libs/SSBA/Apps/bundle_varying

.PHONY : libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/build

libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/requires: libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/bundle_varying.cpp.o.requires

.PHONY : libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/requires

libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/clean:
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/libs/SSBA/Apps && $(CMAKE_COMMAND) -P CMakeFiles/bundle_varying.dir/cmake_clean.cmake
.PHONY : libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/clean

libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/depend:
	cd /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/libs/SSBA/Apps /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/libs/SSBA/Apps /mnt/c/Users/shaya/Documents/copy7-11-3drecon/CTests/build/libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libs/SSBA/Apps/CMakeFiles/bundle_varying.dir/depend

