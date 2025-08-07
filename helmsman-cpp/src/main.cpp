
#include <atomic>
#include <chrono>
#include <csignal>  // For signal handling
#include <iostream>
#include <sstream>  // For std::ostringstream
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// 假设 kcurrent_lib_name 在此编译单元中可用
// 这里我们定义一个临时的，实际项目中可能来自别处
const std::string_view kcurrent_app_name = PROJECT_NAME;

// --- 全局用于信号处理 ---
// static bool g_stop_signal_received = false;
static std::atomic<bool> g_stop_signal_received(false);  // 修改后的代码

void SignalHandler(int signal_num) {
	g_stop_signal_received = true;
	std::ostringstream oss;
	oss << "\nInterrupt signal (" << signal_num << ") received. Shutting down...";
	// 使用 Logger 而不是 cerr
	std::cout << oss.str() << std::endl;
}

bool isDebug() {
	constexpr std::string_view build_type = BUILD_TYPE;
	return build_type == "Debug";
}

bool isRelease() {
	constexpr std::string_view build_type = BUILD_TYPE;
	return build_type == "Release";
}

int main(int argc, char* argv[]) {

	// 1. 设置日志级别 (例如，只记录 Warning 及以上)
	if (isRelease() == true) {
		// logger.setLevel(kinfo);
	} else {
		// logger.setLevel(kdebug);
	}

	// setup signal handler
	signal(SIGINT, SignalHandler);
	// signal(SIGTERM, SignalHandler);

	std::cout << "hello " << kcurrent_app_name << std::endl;

	return 0;
}