from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='cuda_app_layers',
      ext_modules=[CUDAExtension('cuda_app_layers', ['cuda_app_matmul.cu'])],
      cmdclass={'build_ext': BuildExtension})
