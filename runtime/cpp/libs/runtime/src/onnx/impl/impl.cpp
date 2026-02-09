// Copyright (c) 2025 PotterWhite
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// runtime/src/onnx/impl/impl.cpp
#include "impl.h"
#include "Utils/logger/logger.h"

namespace arcforge {
namespace runtime {

auto& logger = arcforge::embedded::utils::Logger::GetInstance();

// #define DEBUG
/*===================================================
 * constructors and operators
 *===================================================*/
Impl::Impl() {
	arcforge::embedded::utils::Logger::GetInstance().Info("Impl object constructed.",
	                                                      kcurrent_lib_name);
}

Impl::~Impl() {
	arcforge::embedded::utils::Logger::GetInstance().Info("Impl cleaned up.", kcurrent_lib_name);
}

void Impl::show_input(const Ort::Session& session) {
	Ort::AllocatorWithDefaultOptions allocator;

	// 4. Echo input message
	size_t num_inputs = session.GetInputCount();
	// std::count << "Number of inputs: " << num_inputs << std::end;
	logger.Info("Number of inputs: " + std::to_string(num_inputs));

	for (size_t i = 0; i < num_inputs; i++) {
		char* input_name = session.GetInputName(i, allocator);
		auto input_type_info = session.GetInputTypeInfo(i);
		auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_shape = tensor_info.GetShape();

		std::cout << "Input " << i << " name: " << input_name << "\n";
		std::cout << "Input shape: [ ";
		for (auto dim : input_shape) {
			std::cout << dim << " ";
		}
		std::cout << "]\n";

		allocator.Free(input_name);
	}
}

void Impl::show_output(const Ort::Session& session) {
	Ort::AllocatorWithDefaultOptions allocator;

	// 5. echo output message
	size_t num_outputs = session.GetOutputCount();
	std::cout << "Number of outputs: " << num_outputs << std::endl;

	for (size_t i = 0; i < num_outputs; i++) {
		char* output_name = session.GetOutputName(i, allocator);
		auto output_type_info = session.GetOutputTypeInfo(i);
		auto tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_shape = tensor_info.GetShape();

		std::cout << "Output " << i << " name: " << output_name << "\n";
		std::cout << "Output shape: [ ";
		for (auto dim : output_shape) {
			std::cout << dim << " ";
		}
		std::cout << "]\n";

		allocator.Free(output_name);
	}
}

Ort::SessionOptions Impl::init_session_option(void) {
	Ort::SessionOptions opt;
	opt.SetIntraOpNumThreads(1);
	opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

	return opt;
}

}  // namespace runtime
}  // namespace arcforge