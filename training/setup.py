from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lut6_cuda',
    ext_modules=[
        CUDAExtension('lut6_cuda', [
            'lut6.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })