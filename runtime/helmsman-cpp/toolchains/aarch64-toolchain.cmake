# toolchains/aarch64-toolchain.cmake
#
# This toolchain file configures CMake for cross-compiling
# to an aarch64 Linux target using a Buildroot SDK.

# I. 设置目标系统信息 (Target System Information)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# II. 设置Sysroot (System Root)
# 这是交叉编译环境的根目录，所有头文件和库都将在这里查找
set(SDK_HOST_PREFIX "/development/docker_volumes/src/sdk/rk3588s-linux/buildroot/output/rockchip_rk3588/host")
set(CMAKE_SYSROOT ${SDK_HOST_PREFIX}/aarch64-buildroot-linux-gnu/sysroot)

# III. 设置交叉编译器 (Cross Compilers)
set(TOOLCHAIN_BIN_DIR "${SDK_HOST_PREFIX}/bin")
set(CMAKE_C_COMPILER ${TOOLCHAIN_BIN_DIR}/aarch64-buildroot-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_BIN_DIR}/aarch64-buildroot-linux-gnu-g++)

# IV. 配置查找路径 (Search Path Configuration)
# 指导CMake只在Sysroot中查找库、头文件和包
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# V. RPATH设置 (可选，但推荐放在这里)
# 让可执行文件在目标板上能找到位于同一目录下的.so文件
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)