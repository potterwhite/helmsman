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
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
