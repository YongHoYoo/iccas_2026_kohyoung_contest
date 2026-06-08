from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "ky_solver._solver",
        ["_solver.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "ky_solver._cy_math",
        ["_cy_math.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "ky_solver._weights",
        ["_weights.pyx"],
        include_dirs=[np.get_include(), "."],
    ),
    Extension(
        "ky_solver.v2_planning",
        ["v2_planning.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="solver",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
