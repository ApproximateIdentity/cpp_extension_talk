import os
from setuptools import setup, Extension

MODULE_EXT = Extension(
    'lrcpp',
    include_dirs = ['.', "./vendor"],
    sources=["lr_exports.cpp", "lr.cpp"],
    extra_compile_args=['-g', '-std=c++11'],
    language='c++')

setup(
    name='lrcpp',
    ext_modules = [MODULE_EXT])
