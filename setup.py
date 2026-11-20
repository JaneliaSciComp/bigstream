import setuptools

setuptools.setup(
    name="bigstream",
    version="5.1.0",
    author="Greg M. Fleishman",
    author_email="greg.nli10me@gmail.com",
    description="Tools for distributed alignment of massive images",
    url="https://github.com/JaneliaScicomp/bigstream",
    license="BSD-3",
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.12',
    install_requires=[
        'numpy>=2',
        'scipy>=1.9.1',
        'opencv-python>=4.5.5.64',
        'bokeh>=2.4.3',
        'dask>=2023.10.1',
        'distributed>=2023.10.1',
        'ClusterWrap>=0.3.0',
        'zarr>=3.2.1,<4',
        'h5py>=3.8.0',
        'numcodecs>=0.16.5',
        'ome-zarr-models>=1.0',
        'fishspot>=0.2.3',
        'SimpleITK>=2.2.0',
        'tifffile>=2025.5.21',
        'morphsnakes>=2.0.0',
        'pynrrd>=1.0.0',
        'scikit-image>=0.20.0',
    ]
)
