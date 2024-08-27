from setuptools import setup, find_packages

setup(
    name='ucurv',
    version='0.1.1',
    author='Duy Nguyen',
    author_email='duyn26364@gmail.com',
    description='uniform discrete curvelet transform',
    # long_description=open('C:\\Users\\duyn2\\projects\\ucurv\\README.md').read(),
    # long_description_content_type='text/markdown',
    install_requires=['numpy',
                       'matplotlib',
                       'pytest'
                       ],
    url='https://github.com/yud08/ucurv',  # URL to the project’s homepage
    # packages=find_packages(),
    packages=['ucurv'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)