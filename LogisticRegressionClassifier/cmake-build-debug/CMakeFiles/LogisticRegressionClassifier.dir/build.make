# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.17

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

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2020.3.1\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2020.3.1\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\agbod\GitHub\Machine_Learning\LogisticRegressionClassifier

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\agbod\GitHub\Machine_Learning\LogisticRegressionClassifier\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/LogisticRegressionClassifier.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/LogisticRegressionClassifier.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LogisticRegressionClassifier.dir/flags.make

CMakeFiles/LogisticRegressionClassifier.dir/main.c.obj: CMakeFiles/LogisticRegressionClassifier.dir/flags.make
CMakeFiles/LogisticRegressionClassifier.dir/main.c.obj: ../main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\agbod\GitHub\Machine_Learning\LogisticRegressionClassifier\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/LogisticRegressionClassifier.dir/main.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\LogisticRegressionClassifier.dir\main.c.obj   -c C:\Users\agbod\GitHub\Machine_Learning\LogisticRegressionClassifier\main.c

CMakeFiles/LogisticRegressionClassifier.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/LogisticRegressionClassifier.dir/main.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\agbod\GitHub\Machine_Learning\LogisticRegressionClassifier\main.c > CMakeFiles\LogisticRegressionClassifier.dir\main.c.i

CMakeFiles/LogisticRegressionClassifier.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/LogisticRegressionClassifier.dir/main.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\agbod\GitHub\Machine_Learning\LogisticRegressionClassifier\main.c -o CMakeFiles\LogisticRegressionClassifier.dir\main.c.s

# Object files for target LogisticRegressionClassifier
LogisticRegressionClassifier_OBJECTS = \
"CMakeFiles/LogisticRegressionClassifier.dir/main.c.obj"

# External object files for target LogisticRegressionClassifier
LogisticRegressionClassifier_EXTERNAL_OBJECTS =

LogisticRegressionClassifier.exe: CMakeFiles/LogisticRegressionClassifier.dir/main.c.obj
LogisticRegressionClassifier.exe: CMakeFiles/LogisticRegressionClassifier.dir/build.make
LogisticRegressionClassifier.exe: CMakeFiles/LogisticRegressionClassifier.dir/linklibs.rsp
LogisticRegressionClassifier.exe: CMakeFiles/LogisticRegressionClassifier.dir/objects1.rsp
LogisticRegressionClassifier.exe: CMakeFiles/LogisticRegressionClassifier.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\agbod\GitHub\Machine_Learning\LogisticRegressionClassifier\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable LogisticRegressionClassifier.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\LogisticRegressionClassifier.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LogisticRegressionClassifier.dir/build: LogisticRegressionClassifier.exe

.PHONY : CMakeFiles/LogisticRegressionClassifier.dir/build

CMakeFiles/LogisticRegressionClassifier.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\LogisticRegressionClassifier.dir\cmake_clean.cmake
.PHONY : CMakeFiles/LogisticRegressionClassifier.dir/clean

CMakeFiles/LogisticRegressionClassifier.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\agbod\GitHub\Machine_Learning\LogisticRegressionClassifier C:\Users\agbod\GitHub\Machine_Learning\LogisticRegressionClassifier C:\Users\agbod\GitHub\Machine_Learning\LogisticRegressionClassifier\cmake-build-debug C:\Users\agbod\GitHub\Machine_Learning\LogisticRegressionClassifier\cmake-build-debug C:\Users\agbod\GitHub\Machine_Learning\LogisticRegressionClassifier\cmake-build-debug\CMakeFiles\LogisticRegressionClassifier.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/LogisticRegressionClassifier.dir/depend

