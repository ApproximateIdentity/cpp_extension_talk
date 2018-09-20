import os
from setuptools import setup, Extension

MODULE_EXT = Extension(
    'module',
    sources=["module.cpp"],
    include_dirs = ["./vendor"],
    extra_compile_args=['-g', '-std=c++11'],
    language='c++')

setup(
    name='module',
    ext_modules = [MODULE_EXT])
