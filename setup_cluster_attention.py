"""
Minimal build of SpargeAttn for ClusterAttention.

Only compiles SM90 kernels and the fused extension.
Requires CUDA >= 12.4 and an SM90 target (H100/H200).

Expects:
  - autogen_cluster_attention.py in csrc/qattn/instantiations_sm90/
    (replacing autogen.py) to generate only the needed instantiations.
  - pybind_cluster_attention.cpp in csrc/qattn/ (registers SM90 only,
    no changes to the original pybind.cpp needed).
"""

import os
import subprocess
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


def get_nvcc_cuda_version(cuda_dir):
    nvcc_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    return parse(output[release_idx].split(",")[0])


def run_instantiations(src_dir):
    for py_file in Path(src_dir).rglob("*.py"):
        print(f"Running: {py_file}")
        os.system(f"python {py_file}")


def get_instantiations(src_dir):
    base_path = Path(src_dir)
    return [
        os.path.join(src_dir, str(path.relative_to(base_path)))
        for path in base_path.rglob("*")
        if path.is_file() and path.suffix == ".cu"
    ]


if CUDA_HOME is None:
    raise RuntimeError("Cannot find CUDA_HOME.")

nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if nvcc_cuda_version < Version("12.4"):
    raise RuntimeError("CUDA 12.4 or higher is required for SM90.")

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

CXX_FLAGS = [
    "-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17",
    "-DENABLE_BF16",
    "-DHAS_SM90",
    f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
]

NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--use_fast_math",
    "--threads=8",
    "-Xptxas=-v",
    "-diag-suppress=174",
    "-Xcompiler", "-include,cassert",
    f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
    "-gencode", "arch=compute_90a,code=sm_90a",
]

# Generate only the SM90 instantiations we need
run_instantiations("csrc/qattn/instantiations_sm90")

qattn_extension = CUDAExtension(
    name="spas_sage_attn._qattn",
    sources=[
        "csrc/qattn/pybind_cluster_attention.cpp",
        "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu",
    ] + get_instantiations("csrc/qattn/instantiations_sm90"),
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
    extra_link_args=["-lcuda"],
)

fused_extension = CUDAExtension(
    name="spas_sage_attn._fused",
    sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)

setup(
    name="spas_sage_attn",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[qattn_extension, fused_extension],
    cmdclass={"build_ext": BuildExtension},
)