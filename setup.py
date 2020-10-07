from setuptools import setup, find_packages

setup(
    name='uobrainflex',
    version='0.1dev',
    author='Santiago Jaramillo',
    author_email='sjara@uoregon.edu',
    description='Basic tools for the UO project on brain states and flexible behavior.',
    packages=find_packages(exclude=[]),
    long_description=open('README.md').read(),
    long_description_content_type='text/x-rst; charset=UTF-8',
    install_requires=['datajoint>=0.12.5','pynwb>=1.4']
)


