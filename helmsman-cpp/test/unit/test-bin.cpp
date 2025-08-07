// app-first/main.cpp

// ... 其他 include ...
#include "main-utils.h"  // 包含头文件

// ... kcurrent_app_name, SignalHandler 等代码保持不变 ...

// 函数定义
bool isDebug() {
	constexpr std::string_view build_type = BUILD_TYPE;
	return build_type == "Debug";
}

bool isRelease() {
	constexpr std::string_view build_type = BUILD_TYPE;
	return build_type == "Release";
}

// ... main 函数保持不变 ...