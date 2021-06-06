from setuptools import setup, Extension

setup(
    name='mykmeanssp',
    version='0.0.1',
    author='Ben_&_Ofek',
    description='connecting kmeans algoritim from C to python program',
    ext_modules=[Extension('mykmeanssp', ['kmeans.c'])]
)


