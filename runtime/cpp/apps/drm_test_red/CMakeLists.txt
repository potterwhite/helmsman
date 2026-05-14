# Copyright (c) 2026 PotterWhite
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

# apps/drm_test_red/CMakeLists.txt — minimal standalone DRMKit smoke test

if(NOT DEFINED GLOBAL_VERSION_STRING OR "${GLOBAL_VERSION_STRING}" STREQUAL "")
    set(GLOBAL_VERSION_STRING "99.99.99")
endif()

set(PROJECT_NAME "drm_test_red")
project(${PROJECT_NAME}
    VERSION ${GLOBAL_VERSION_STRING}
    LANGUAGES CXX
)

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/drm_test_red.cpp
)

if(NOT DEFINED PROJECT_NAMESPACE)
    set(PROJECT_NAMESPACE "ArcForge")
endif()

arc_setup_system_info(${PROJECT_NAME})

if(NOT TARGET ${PROJECT_NAMESPACE}::DRMKit)
    find_package(${PROJECT_NAMESPACE}_DRMKit REQUIRED)
endif()

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${PROJECT_NAMESPACE}::DRMKit
)

arc_install_executable(${PROJECT_NAME})
