// Copyright (c) 2026 PotterWhite
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

#include "pipeline/pipeline.h"
#include "common/common-define.h"
#include "pipeline/stages/inference-engine/inference-engine-factory.h"

using helmsman::utils::timing::ScopedTimer;

void Pipeline::SetTimingEnabled(bool enabled) { config_.timing_enabled = enabled; }
bool Pipeline::IsTimingEnabled() const { return config_.timing_enabled; }

std::unique_ptr<InferenceEngine> Pipeline::MakeEngine() {
	return createInferenceEngine();
}

Pipeline& Pipeline::GetInstance() {
	static Pipeline instance;
	return instance;
}

Pipeline::Pipeline() {
	helmsman::utils::Logger::GetInstance().Info("Pipeline object constructed.",
	                                            kcurrent_module_name);
}

Pipeline::~Pipeline() {
	helmsman::utils::Logger::GetInstance().Info("Pipeline cleaned up.", kcurrent_module_name);
}

void Pipeline::VerifyParametersNecessary() {
	if (!config_.is_video && config_.input_path.empty()) {
		throw std::invalid_argument("No input source: neither video nor image provided.");
	}
	if (config_.model_path.empty()) {
		throw std::invalid_argument("Model path is empty.");
	}
	if (config_.output_bin_path.empty()) {
		throw std::invalid_argument("Output binary path is empty.");
	}
}

void Pipeline::Init(const AppConfig& config) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	config_ = config;

	// 1. Frontend (video only)
	if (config_.is_video) {
		try {
			frontend_ = FrontendBase::Create(config_);
		} catch (const std::exception& e) {
			logger.Error(std::string("Failed to create Frontend: ") + e.what(),
			             kcurrent_module_name);
			throw;
		}

		logger.Info("Video source: " + std::to_string(frontend_->width()) + "x" +
		                std::to_string(frontend_->height()) + " @ " +
		                std::to_string(frontend_->fps()) + " fps",
		            kcurrent_module_name);

		if (config_.model_type == ModelType::kMODNet) {
			logger.Error("MODNet does not support video input. Use --rvm for video.",
			             kcurrent_module_name);
			throw std::invalid_argument("MODNet does not support video input.");
		}
	}

	// 2. InferenceEngine + model load
	engine_ = MakeEngine();
	engine_->SetAppConfig(config_);
	{
		ScopedTimer t("Pipeline::Init() model load", config_.timing_enabled, logger,
		              kcurrent_module_name);
		engine_->Load(config_.model_path);
	}

	// 3. Backend configuration
	backend_.SetAppConfig(config_);

	// 4. Inject dependencies into modes
	rvm_mode_.SetAppConfig(config_);
	rvm_mode_.SetFrontend(frontend_.get());
	rvm_mode_.SetEngine(engine_.get());
	rvm_mode_.SetBackend(&backend_);

	modnet_mode_.SetEngine(engine_.get());
	modnet_mode_.SetBackend(&backend_);
	modnet_mode_.SetAppConfig(config_);
}

int Pipeline::Run() {
	VerifyParametersNecessary();

	auto& logger = helmsman::utils::Logger::GetInstance();

	switch (config_.model_type) {
		case ModelType::kMODNet:
			logger.Info("Pipeline: running MODNet path (single-frame)", kcurrent_module_name);
			return modnet_mode_.Run();
		case ModelType::kRVM:
			logger.Info("Pipeline: running RVM path (recurrent multi-frame)", kcurrent_module_name);
			return rvm_mode_.Run();
		default:
			throw std::runtime_error("Unknown model type");
	}
}