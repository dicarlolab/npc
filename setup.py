from distutils.core import setup

setup(
    name='NeuralPopulationControl',
    version='0.1',
    packages=['npc'],
    install_requires=['numpy', 'scipy', 'h5py', 'tensorflow-gpu', 'absl-py', 'pillow'],
    url='https://github.com/dicarlolab/npc.git',
    license='MIT License',
    author='Pouya Bashivan',
    description="Stimulus generator for neural population control."
)
