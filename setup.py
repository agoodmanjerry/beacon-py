from setuptools import setup, Extension
import numpy

# C extension for Burg's AR estimation algorithm
burg_module = Extension(
    'beacon._burg',  # Module will be imported as: from beacon._burg import burg
    sources=[
        'beacon/_burg/burg.c',
        'beacon/_burg/burgmodule.c'
    ],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O2', '-fno-fast-math', '-ffloat-store']
)

setup(
    ext_modules=[burg_module]
)
