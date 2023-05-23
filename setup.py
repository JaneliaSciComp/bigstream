import setuptools

setuptools.setup(
    name="bigstream",
    version="0.0.2",
    author="Greg M. Fleishman",
    author_email="greg.nli10me@gmail.com",
    description="Tools for distributed alignment of massive images",
    url="https://github.com/GFleishman/bigstream",
    license="MIT",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'scikit-image',
        'opencv-python',
        'greedypy',
        'zarr',
        'numcodecs',
    ]
)
