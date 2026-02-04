# Copyright (c) 2025 PotterWhite
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ==============================================================================
# ArcForge Helper Functions
# ==============================================================================

# ------------------------------------------------------------------------------
# Function: arc_init_project_metadata
# Description:
#   Initializes project identity information such as versioning, author info,
#   and build timestamp.
#   It exports these variables to the PARENT_SCOPE so they are globally accessible.
# ------------------------------------------------------------------------------
function(arc_init_project_metadata)

    # 1. Versioning
    # (Relies on project() being called beforehand)
    set(GLOBAL_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}" PARENT_SCOPE)
    set(GLOBAL_VERSION_MINOR "${PROJECT_VERSION_MINOR}" PARENT_SCOPE)
    set(GLOBAL_VERSION_PATCH "${PROJECT_VERSION_PATCH}" PARENT_SCOPE)
    set(GLOBAL_VERSION_STRING "${PROJECT_VERSION}" PARENT_SCOPE)
    set(GLOBAL_EXTENDED_VERSION_STRING "${PROJECT_NAMESPACE} v${PROJECT_VERSION}" PARENT_SCOPE)

    # 2. Author Information
    set(GLOBAL_AUTHOR_NAME "PotterWhite" PARENT_SCOPE)
    set(GLOBAL_AUTHOR_EMAIL "themanuknowwhom@outlook.com" PARENT_SCOPE)

    # 3. Build Timestamp
    # "UTC" ensures reproducibility across timezones
    string(TIMESTAMP _CURRENT_TIMESTAMP "%Y-%m-%d %H:%M:%S UTC")

    # Export the local variable to parent scope
    set(GLOBAL_BUILD_TIMESTAMP "${_CURRENT_TIMESTAMP}" PARENT_SCOPE)
    set(GLOBAL_EXTENDED_BUILD_TIMESTAMP_STRING "${_CURRENT_TIMESTAMP}" PARENT_SCOPE)

    message(STATUS "[Configuration] Metadata initialized: ${PROJECT_NAMESPACE} v${PROJECT_VERSION} (@${_CURRENT_TIMESTAMP})")

endfunction()

# ------------------------------------------------------------------------------
# Function: arc_init_global_settings
# Description:
#   Initializes the global build configuration for the entire project.
#   Instead of polluting the global namespace with flags, it encapsulates
#   settings into an Interface Library ('arc_base_settings').
#
#   It performs the following actions:
#     1. Creates the 'arc_base_settings' interface target.
#     2. Enforces C++17 standard and Position Independent Code (PIC/PIE).
#     3. Applies a strict list of compiler warnings (Warnings as Errors).
#     4. Detects/Defaults CMAKE_BUILD_TYPE and sets optimization flags (-O3, -g).
#     5. Exports build type variables to the PARENT_SCOPE for install path usage.
#
# Arguments:
#   None
# ------------------------------------------------------------------------------
function(arc_init_global_settings)

    # --------------------------------------------------------------------------
    # 1. Create Interface Target
    # All sub-libraries will link to this to inherit settings.
    # --------------------------------------------------------------------------
    set(ARC_SETTING_INTERFACE_NAME "arc_base_settings")
    add_library(${ARC_SETTING_INTERFACE_NAME} INTERFACE)

    # --------------------------------------------------------------------------
    # 2. Basic Properties (Standard & PIC)
    # --------------------------------------------------------------------------

    # Enforce C++17
    target_compile_features(${ARC_SETTING_INTERFACE_NAME}
        INTERFACE
            cxx_std_17
    )

    # Position Independent Code (PIC)
    #    - Shared Libs: Requires PIC (CMake handles this default, but explicit is safe)
    #    - Static Libs: Force PIC so they can be linked into Shared Libs later
    #    - Executables: Enables PIE (Security feature)
    set_target_properties(${ARC_SETTING_INTERFACE_NAME}
        PROPERTIES
            INTERFACE_POSITION_INDEPENDENT_CODE
                ON
    )

    # --------------------------------------------------------------------------
    # 3. Build Type Logic & Optimization
    # --------------------------------------------------------------------------
    # Set default build type to "Release" if not specified
    if(NOT CMAKE_BUILD_TYPE)
        set(TYPE_VAR "Release")
    else()
        set(TYPE_VAR "${CMAKE_BUILD_TYPE}")
    endif()

    # Normalize to lowercase for directory naming (e.g., install/release)
    string(TOLOWER "${TYPE_VAR}" TYPE_VAR_LOWER)

    # Export variables to Parent Scope (Crucial for Root CMake install logic)
    set(FINAL_BUILD_TYPE "${TYPE_VAR}" PARENT_SCOPE)
    set(BUILD_TYPE_LOWER "${TYPE_VAR_LOWER}" PARENT_SCOPE)

    # Inject macro definition for code usage
    target_compile_definitions(${ARC_SETTING_INTERFACE_NAME} INTERFACE BUILD_TYPE="${TYPE_VAR}")

    # --------------------------------------------------------------------------
    # 4. Compile Policy Select
    # --------------------------------------------------------------------------

    set(BASIC_COMPILE_OPTIONS
        -Wall -Wextra -Werror
    )

    set(STRINGENT_COMPILE_OPTIONS
        -Wconversion
        -Wsign-conversion
        -Wfloat-conversion
        -Wpedantic
        -Wcast-qual
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wformat=2
        -Wformat-security
        -Wformat-nonliteral
        -Wuninitialized
        -Winit-self
        -Wswitch-enum
        -Wswitch-default
        -Wmissing-include-dirs
        -Wredundant-decls
        -Wshadow
        -Wundef
        -Wdouble-promotion
    )

    # Apply Options based on Build Type
    # Note: I included ${BASIC_COMPILE_OPTIONS} here to ensure -Wall/-Werror are applied.
    if(TYPE_VAR STREQUAL "Debug")
        # Debug: No optimization, with debug info
        target_compile_options(${ARC_SETTING_INTERFACE_NAME} INTERFACE
            -Og -g -DDEBUG
            ${BASIC_COMPILE_OPTIONS}
            ${STRINGENT_COMPILE_OPTIONS}
        )
    elseif(TYPE_VAR STREQUAL "RelWithDebInfo")
        # RelWithDebInfo: Full optimization with debug info
        target_compile_options(${ARC_SETTING_INTERFACE_NAME} INTERFACE
            -O2 -g -DNDEBUG
            ${BASIC_COMPILE_OPTIONS}
            ${STRINGENT_COMPILE_OPTIONS}
        )
    elseif(TYPE_VAR STREQUAL "MinSizeRel")
        # MinSizeRel: Size optimization
        target_compile_options(${ARC_SETTING_INTERFACE_NAME} INTERFACE
            -Os -DNDEBUG
            ${BASIC_COMPILE_OPTIONS}
            ${STRINGENT_COMPILE_OPTIONS}
        )
    else()
        # Release: Full optimization, no debug info
        target_compile_options(${ARC_SETTING_INTERFACE_NAME} INTERFACE
            -O3 -DNDEBUG
            ${BASIC_COMPILE_OPTIONS}
            ${STRINGENT_COMPILE_OPTIONS}
        )
    endif()

    message(STATUS "[Configuration] Global settings initialized. Build Type: ${TYPE_VAR}")

endfunction()


# ------------------------------------------------------------------------------
# Function: arc_setup_system_info
# Description:
#   A centralized function to configure standard build settings for sub-libraries.
#   It performs the following actions:
#     1. Generates 'system-info.h' from the global template.
#     2. Adds include directories (Source include + Generated include).
#     3. Convention: If "<TargetTopDir>/include" directory exists, use it.
#     4. Convention: If include/<TargetName>/pch.h exists, use it.
#     5. Sets target properties (VERSION, SOVERSION, OUTPUT_NAME).
#     6. Injects the PROJECT_NAME macro definition.
#     7. Links against the common configuration target "arc_base_settings".
#     8. Configures installation rules for the generated header.
#
# Arguments:
#   target_name : The specific library target name (e.g., utils, network).
# ------------------------------------------------------------------------------
function(arc_setup_system_info target_name)

    # --------------------------------------------------------------------------
    # 0. Integrity Check
    # Ensure required global variables are defined in the root CMakeLists.txt.
    # --------------------------------------------------------------------------
    if(NOT DEFINED GLOBAL_EXTENDED_VERSION_STRING)
        message(FATAL_ERROR "[Configuration] Error: Global version variables are missing. Please define them in the root CMakeLists.txt.")
    endif()

    if(NOT DEFINED PROJECT_NAMESPACE)
        message(FATAL_ERROR "[Configuration] Error: PROJECT_NAMESPACE is not defined.")
    endif()

    # --------------------------------------------------------------------------
    # 1. Prepare Variables for Template Substitution
    # The .in template uses @ARG_TARGET_NAME@ to identify the specific component.
    # --------------------------------------------------------------------------
    set(ARG_TARGET_NAME "${target_name}")

    get_target_property(TARGET_TYPE ${target_name} TYPE)

    # --------------------------------------------------------------------------
    # 2. Configure Header File
    # Generate system-info.h inside the build directory to avoid polluting source.
    # --------------------------------------------------------------------------
    set(TEMPLATE_FILE "${CMAKE_SOURCE_DIR}/cmake/templates/system-info.h.in")

    # Output path: build/generated_headers/<Namespace>/<Target>/system-info.h
    # We include ${target_name} in the path to prevent file collisions between libraries.
    set(HEADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_headers/${PROJECT_NAMESPACE}/${target_name}")
    set(HEADER_OUTPUT_FILE "${HEADER_OUTPUT_DIR}/system-info.h")

    if(EXISTS "${TEMPLATE_FILE}")
        configure_file("${TEMPLATE_FILE}" "${HEADER_OUTPUT_FILE}" @ONLY)
    else()
        message(FATAL_ERROR "[Configuration] Template file not found: ${TEMPLATE_FILE}")
    endif()

    # --------------------------------------------------------------------------
    # 3. Setup Include Directories
    #    Expose the generated header directory to the target and its consumers.
    #    Convention: If "<TargetTopDir>/include" directory exists, use it.
    # --------------------------------------------------------------------------

    # CMAKE_CURRENT_SOURCE_DIR is not current .cmake`s Path but it is the Path of whom invokes current .cmake file.
    set(DEFAULT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

    target_include_directories(${target_name} PUBLIC
        # For build tree: Point to the generated directory
        "$<BUILD_INTERFACE:${HEADER_OUTPUT_DIR}>"

        "$<BUILD_INTERFACE:${DEFAULT_INCLUDE_DIR}>"

        # For install tree: Point to standard include directory
        "$<INSTALL_INTERFACE:include>"
    )

    # --------------------------------------------------------------------------
    # 4. pch(PreCompile Header)
    #    Convention: If include/<TargetName>/pch.h exists, use it.
    # --------------------------------------------------------------------------
    set(DEFAULT_PCH_FILE "${DEFAULT_INCLUDE_DIR}/${target_name}/pch.h")
    if(EXISTS "${DEFAULT_PCH_FILE}")
        # within this if branch to increase the robust
        target_precompile_headers(${target_name}
            PRIVATE
                ${DEFAULT_PCH_FILE}
        )
    endif()

    # --------------------------------------------------------------------------
    # 5. Set Target Properties (Version Control)
    # Apply standard versioning to shared libraries (.so).
    # --------------------------------------------------------------------------

    set_target_properties(${target_name}
        PROPERTIES
            OUTPUT_NAME
                "${PROJECT_NAMESPACE}_${target_name}"   # format output name
    )

    if(TARGET_TYPE STREQUAL "SHARED_LIBRARY")
        set_target_properties(${target_name}
            PROPERTIES
                VERSION
                    "${PROJECT_VERSION}"                    # e.g., 1.0.1
                SOVERSION
                    "${PROJECT_VERSION_MAJOR}"              # e.g., 1
        )
    endif()

    # --------------------------------------------------------------------------
    # 6. Inject Compile Definitions
    # Allows C++ code to use the PROJECT_NAME macro without extra includes.
    # --------------------------------------------------------------------------
    target_compile_definitions(${target_name}
        PRIVATE
            PROJECT_NAME="${target_name}"
    )

    # --------------------------------------------------------------------------
    # 7. Do Several Important settings
    #    C++ standard/PIC/BuildType/CompilePolicy
    # --------------------------------------------------------------------------
    target_link_libraries(${target_name}
        PRIVATE
            $<BUILD_INTERFACE:arc_base_settings>
    )

    # --------------------------------------------------------------------------
    # 8. Install Rules
    # Install the generated header to include/ArcForge/<Target>/
    # ** Why it has been cancelled? **
    #    1st. Because system-infor.h is common and internal file
    #         There is no need to export to our users.
    #    2nd.
    #         <TargetTopDir>/include default to external & public include dir
    #         so no matter here we install or not the system-info.h will be synchronized and exported to final users
    #    ***
    # ** Leave these code for future expand usage**
    # --------------------------------------------------------------------------
    #[[
    install(FILES "${HEADER_OUTPUT_FILE}"
            DESTINATION "include/${PROJECT_NAMESPACE}/${target_name}"
            COMPONENT headers
    )
    ]]

    message(STATUS "[Configuration] Configured system info for target: ${target_name}")

endfunction()


# ------------------------------------------------------------------------------
# Function: arc_install_executable
# Description:
#   A simplified installation function specifically for executable applications.
#   Unlike libraries, executables typically do not require header exports or
#   complex CMake config files for consumers.
#   It performs:
#     1. Installs the runtime binary (executable) to the 'bin' directory.
#
# Arguments:
#   target_name : The executable target name (created via add_executable).
# ------------------------------------------------------------------------------
function(arc_install_executable target_name)
    install(TARGETS ${target_name}
        RUNTIME DESTINATION bin
    )
    message(STATUS "[Configuration] Configured install rules for App: ${target_name}")
endfunction()

# ------------------------------------------------------------------------------
# Function: arc_install_library
# Description:
#   A centralized function to handle the complex installation logic for libraries.
#   It performs:
#     1. Installs physical headers (Generic directory install).
#     2. Installs physical binaries (.so, .a) and defines EXPORT set.
#     3. Generates and installs CMake Config/Version files (The "Passport").
#     4. Generates and installs Targets file.
#
# Arguments:
#   target_name : The library target name (e.g., Utils).
#   include_dir : The directory containing public headers (e.g., ${INCLUDE_DIR}).
# ------------------------------------------------------------------------------
function(arc_install_library target_name include_dir)

    # 0. Dependencies
    include(CMakePackageConfigHelpers)

    # 1. Integrity Check
    if(NOT DEFINED PROJECT_NAMESPACE)
        message(FATAL_ERROR "[Configuration] Error: PROJECT_NAMESPACE is missing.")
    endif()
    # Use PROJECT_VERSION to avoid "No VERSION specified" error in sub-projects
    if(NOT DEFINED PROJECT_VERSION)
        message(FATAL_ERROR "[Configuration] Error: PROJECT_VERSION is missing.")
    endif()

    # 2. Prepare Variables
    # Define the full package name (e.g., ArcForge_Utils)
    set(FULL_PACKAGE_NAME "${PROJECT_NAMESPACE}_${target_name}")
    set(INSTALL_CONFIG_DIR "lib/cmake/${FULL_PACKAGE_NAME}")

    # 3. Install Headers (Replaces the specific pch.h install with generic folder install)
    if(EXISTS "${include_dir}")
        install(
            DIRECTORY "${include_dir}/"
            DESTINATION include
        )
    else()
        message(WARNING "[Configuration] Warning: Public include dir not found: ${include_dir}")
    endif()

    # 4. Install Targets
    install(
        TARGETS
            ${target_name}
        EXPORT
            ${target_name}
        LIBRARY DESTINATION         # so files
            lib
        ARCHIVE DESTINATION         # a files
            lib
        RUNTIME DESTINATION         # bin files
            bin
        INCLUDES DESTINATION        # Tells consumers where headers are
            include
    )

    # 5. Generate ConfigVersion.cmake (tell find_package the version code)
    set(VERSION_FILE_NAME "${FULL_PACKAGE_NAME}ConfigVersion.cmake")
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/${VERSION_FILE_NAME}"
        VERSION ${PROJECT_VERSION} # Used Global Version to fix scope issue
        COMPATIBILITY SameMajorVersion
    )

    # 6. Generate Config.cmake (The entrance)
    #    Create a simple Config template, tell cmake the place where it can find Targets files.
    set(CONFIG_FILE_NAME "${FULL_PACKAGE_NAME}Config.cmake")
    set(CONFIG_TEMPLATE_FILE "${CMAKE_CURRENT_BINARY_DIR}/${FULL_PACKAGE_NAME}Config.cmake.in")

    #    (Using file WRITE to avoid creating physical .in files in source tree)
    file(WRITE "${CONFIG_TEMPLATE_FILE}"
        "@PACKAGE_INIT@\n"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/${FULL_PACKAGE_NAME}Targets.cmake\")\n"
        "check_required_components(${target_name})\n"
    )

    #    Produce final Config.cmake according to above template
    configure_package_config_file(
        "${CONFIG_TEMPLATE_FILE}"
        "${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}"
        INSTALL_DESTINATION "${INSTALL_CONFIG_DIR}"
    )

    # 7. Install Config.cmake & ConfigVersion.cmake who generated above (Build Dir -> Install Dir)
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}"
        "${CMAKE_CURRENT_BINARY_DIR}/${VERSION_FILE_NAME}"
        DESTINATION "${INSTALL_CONFIG_DIR}"
    )

    # 8. Install Export list (Targets.cmake)
    #    It will sync all Target information into files
    install(
        EXPORT ${target_name}                                       # <-- Tell CMake which export set we want to install
        FILE "${FULL_PACKAGE_NAME}Targets.cmake"                    # Name of the generated .cmake file
        NAMESPACE "${PROJECT_NAMESPACE}::"                          # (Recommended) Add a namespace to targets to avoid conflicts
        DESTINATION "${INSTALL_CONFIG_DIR}"                         # Standard installation path for config files
    )

    message(STATUS "[Configuration] Configured install rules for lib: ${target_name}")

endfunction()

# ------------------------------------------------------------------------------
# Function: arc_add_test
# Description:
#   A standardized function to register a GoogleTest unit test for a specific module.
#   It automates the following boilerplate tasks:
#     1. Creates an executable named "test_<TargetName>".
#     2. Links the GoogleTest main entry point (GTest::gtest_main).
#     3. Links the target library (e.g., ArcForge::Network).
#     4. Injects the library's internal 'src' directory into the include path
#        (Enabling white-box testing of private headers).
#     5. Registers the test binary with CTest via gtest_discover_tests().
#
# Arguments:
#   target_name : The name of the library being tested (e.g., "Network").
#                 NOTE: Do not add the namespace prefix here; the function adds
#                 "${PROJECT_NAMESPACE}::" automatically.
#   ARGN        : The list of source files for the test (e.g., "test_network.cpp").
# ------------------------------------------------------------------------------
function(arc_add_test target_name)
    # 0. Parsing Arguments
    # The first argument is target_name; the rest are source files.
    set(test_sources ${ARGN})

    # 1. Define Test Executable Name
    # Convention: test_<LibraryName> (e.g., test_Network)
    set(test_exe_name "test_${target_name}")

    message(STATUS "[Testing] Configuring test target: ${test_exe_name} for module: ${target_name}")

    # 2. Create Executable
    add_executable(${test_exe_name} ${test_sources})

    # 3. Link GoogleTest
    # GTest::gtest_main includes the standard main() function.
    # GTest::gtest      includes the testing framework.
    target_link_libraries(${test_exe_name}
        PRIVATE
            GTest::gtest
            GTest::gtest_main
    )

    # 4. Link the Target Module
    # We assume the library is aliased as ${PROJECT_NAMESPACE}::<TargetName>
    # (e.g., ArcForge::Network).
    # We check if the target exists to provide a friendly error message.
    set(aliased_target "${PROJECT_NAMESPACE}::${target_name}")

    if(TARGET ${aliased_target})
        target_link_libraries(${test_exe_name}
            PRIVATE
                ${aliased_target}
        )
    else()
        message(FATAL_ERROR "[Testing] Error: Cannot link test '${test_exe_name}' against unknown target '${aliased_target}'. Ensure the library is defined before adding tests.")
    endif()

    # 5. Enable White-box Testing (Internal Includes)
    # Unit tests often need access to private headers located in 'src/'.
    # We retrieve the SOURCE_DIR of the original library target and append '/src'.
    # Note: We must resolve the ALIAS to the actual target name first (e.g., ArcForge::Network -> Network).
    get_target_property(actual_target_name ${aliased_target} ALIASED_TARGET)

    if(NOT actual_target_name)
        # If it's not an alias, use the name directly
        set(actual_target_name ${aliased_target})
    endif()

    get_target_property(target_source_dir ${actual_target_name} SOURCE_DIR)

    if(EXISTS "${target_source_dir}/src")
        target_include_directories(${test_exe_name}
            PRIVATE
                "${target_source_dir}/src"
        )
    endif()

    # 6. Register with CTest
    # This parses the compiled binary to list all tests for CTest.
    gtest_discover_tests(${test_exe_name}
        # Optional: Add XML output for CI/CD parsers (like Jenkins/GitLab)
        XML_OUTPUT_DIR "${CMAKE_BINARY_DIR}/test_results"
    )

    # ---------------------------------
    # IV. (Optional) Fix "PROJECT_NAME macro is missing"
    # ---------------------------------
    # In test cpp code, if desire to use #ifdef PROJECT_NAME to check
    # whether the macro is defined during compilation of the test program.
    # This is actually checking if the macro is defined when building the test.
    # The arc_add_test function may not add this macro by default.
    # If you really want that test to pass, you can manually add it here:
    target_compile_definitions(${test_exe_name}
        PRIVATE
            PROJECT_NAME="${PROJECT_NAME}"
    )

endfunction()

# ------------------------------------------------------------------------------
# Function: arc_extract_version_from_changelog
# Description:
#   Reads CHANGELOG.md, extracts the latest version (format ## [vX.Y.Z]),
#   and sets the output variable in the PARENT_SCOPE.
# Usage:
#   arc_extract_version_from_changelog(MY_VERSION_VAR)
# ------------------------------------------------------------------------------
function(arc_extract_version_from_changelog output_version_var)
    set(changelog_path "${CMAKE_SOURCE_DIR}/CHANGELOG.md")
    set(default_version "0.0.0")

    if(EXISTS "${changelog_path}")
        # Read first 4KB to save time
        file(READ "${changelog_path}" changelog_content LIMIT 4096)

        # Regex to match: ## [v1.2.3] or ## [1.2.3]
        # Group 1 will be the version number
        string(REGEX MATCH "## \\[v?([0-9]+\\.[0-9]+\\.[0-9]+)\\]" _match_result "${changelog_content}")

        if(CMAKE_MATCH_1)
            set(detected_version "${CMAKE_MATCH_1}")
            message(STATUS "[Version] Extracted from CHANGELOG.md: ${detected_version}")
        else()
            message(WARNING "[Version] Pattern '## [vX.Y.Z]' not found in CHANGELOG.md. Using default.")
            set(detected_version "${default_version}")
        endif()
    else()
        message(WARNING "[Version] CHANGELOG.md not found at ${changelog_path}. Using default.")
        set(detected_version "${default_version}")
    endif()

    # Return the value to the caller
    set(${output_version_var} "${detected_version}" PARENT_SCOPE)
endfunction()


# ------------------------------------------------------------------------------
# Function: arc_generate_system_info_header
# Description:
#   Generates a C++ header file with version and build timestamp info.
#   It creates an interface library 'project_version_info' for easy linking.
# Usage:
#   arc_generate_system_info_header()
# ------------------------------------------------------------------------------
function(arc_generate_system_info_header)
    # Get Timestamp
    string(TIMESTAMP BUILD_TIMESTAMP "%Y-%m-%d %H:%M:%S")

    # Define paths
    set(TEMPLATE_FILE "${CMAKE_SOURCE_DIR}/cmake/templates/system-info.h.in")
    set(GENERATED_DIR "${CMAKE_BINARY_DIR}/generated/include")
    set(OUTPUT_FILE   "${GENERATED_DIR}/ArcForge/system-info-gen.h")

    if(NOT EXISTS "${TEMPLATE_FILE}")
        message(FATAL_ERROR "[Version] Template file not found: ${TEMPLATE_FILE}")
    endif()

    # Configure file (Replace @VARS@)
    configure_file(
        "${TEMPLATE_FILE}"
        "${OUTPUT_FILE}"
        @ONLY
    )

    # Create an Interface Library
    # This allows other libs to just target_link_libraries(target PUBLIC project_version_info)
    if(NOT TARGET project_version_info)
        add_library(project_version_info INTERFACE)
        target_include_directories(project_version_info INTERFACE "${GENERATED_DIR}")
        message(STATUS "[Version] Generated system info header at: ${OUTPUT_FILE}")
    endif()
endfunction()
