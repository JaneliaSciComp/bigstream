import setuptools

setuptools.setup(
    name="bigstream",
    version="1.1.1",
    author="Greg M. Fleishman",
    author_email="greg.nli10me@gmail.com",
    description="Tools for distributed alignment of massive images",
    url="https://github.com/GFleishman/bigstream",
    license="MIT",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.20.3',
        'scipy>=1.9.1',
        'opencv-python>=4.5.5.64',
        'bokeh>=2.4.3',
        'dask>=2022.9.1',
        'dask[array]>=2022.9.1',
        'dask[delayed]>=2022.9.1',
        'dask[distributed]>=2022.9.1',
        'ClusterWrap>=0.3.0',
        'zarr>=2.12.0',
        'numcodecs>=0.9.1',
        'fishspot>=0.2.3',
        'SimpleITK>=2.2.0',
        'tifffile>=2022.10.10'
    ]
)
