from setuptools import setup, find_packages
with open('./requirements.txt', 'r') as r:
    req = r.read().splitlines()

setup(
    name='qunum',
    version='1.0',
    description='''This is the physik package for PINN's, Quantum Numerics and Quantum Symbolic Calculations.''',
    author='Matthew Riccio',
    author_email='riccio_matt@outlook.com',
    packages=find_packages(exclude='test'),  
    install_requires=req,
    include_package_data = True
)

