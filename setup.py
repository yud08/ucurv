from setuptools import setup, find_packages

setup(
    name='ucurv',
    version='0.1.0',
    author='Duy Nguyen',
    author_email='duyn26364@gmail.com',
    description='uniform discrete curvelet transform',
    long_description=open('C:\\Users\\duyn2\\projects\\ucurv\\README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yud08/ucurv',  # URL to the project’s homepage
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)