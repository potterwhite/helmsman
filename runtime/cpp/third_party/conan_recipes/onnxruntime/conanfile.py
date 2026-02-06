from conan import ConanFile
from conan.tools.files import get, copy
import os

class OnnxRuntimeConan(ConanFile):
    name = "onnxruntime"
    description = "Microsoft ONNX Runtime prebuilt binaries"
    settings = "os", "arch", "compiler", "build_type"

    def set_version(self):
        """
        The version is provided externally, typically from a Python venv:
        export ONNXRUNTIME_VERSION=$(python -c "import onnxruntime as ort; print(ort.__version__)")
        """
        ort_version = os.getenv("ONNXRUNTIME_VERSION")
        if not ort_version:
            raise RuntimeError(
                "ONNXRUNTIME_VERSION environment variable is not set. "
                "Please export ONNXRUNTIME_VERSION before running Conan."
            )
        self.version = ort_version

    def package_id(self):
        # This is a prebuilt binary package, ignore compiler compatibility
        self.info.clear()

    def source(self):
        version = self.version
        url = (
            f"https://github.com/microsoft/onnxruntime/releases/download/"
            f"v{version}/onnxruntime-linux-x64-{version}.tgz"
        )

        self.output.info(f"Downloading ONNX Runtime {version}")
        get(
            self,
            url=url,
            strip_root=True
        )

    def package(self):
        copy(
            self,
            pattern="*.h",
            src=os.path.join(self.source_folder, "include"),
            dst=os.path.join(self.package_folder, "include"),
        )

        copy(
            self,
            pattern="libonnxruntime.so*",
            src=os.path.join(self.source_folder, "lib"),
            dst=os.path.join(self.package_folder, "lib"),
        )

    def package_info(self):
        self.cpp_info.libs = ["onnxruntime"]
        # Key & Important: Tell CMakeDeps to generate target
        self.cpp_info.set_property("cmake_file_name", "onnxruntime")
        self.cpp_info.set_property("cmake_target_name", "onnxruntime::onnxruntime")
