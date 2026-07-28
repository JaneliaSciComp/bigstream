# ---- builder: compile SimpleITK 3.0.0b1 with Elastix bundled ----
# SimpleITK_USE_ELASTIX is off by default, and a from-source build of this
# same git ref without the flag was confirmed to be missing
# ElastixImageFilter, so it must be built explicitly rather than relying on
# a prebuilt wheel from --find-links.
FROM ghcr.io/janeliascicomp/dask:2026.6.0-py12-ol9 AS simpleitk-builder

RUN dnf -y groupinstall "Development Tools" && \
    dnf install -y \
        git \
        make \
        cmake

RUN CMAKE_ARGS="-DSimpleITK_USE_ELASTIX=ON" CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
    pip wheel --no-deps -w /wheels \
        "SimpleITK @ git+https://github.com/SimpleITK/SimpleITK@v3.0.0b1"

# ---- final image ----
FROM ghcr.io/janeliascicomp/dask:2026.6.0-py12-ol9

RUN dnf install -y \
        git \
        mesa-libGL \
        libzstd-devel

WORKDIR /app/bigstream

ENV ITK_THREADS=
ENV MKL_NUM_THREADS=
ENV NUM_MKL_THREADS=
ENV OPENBLAS_NUM_THREADS=
ENV OPENMP_NUM_THREADS=
ENV OMP_NUM_THREADS=

ENV PYTHONPATH=/app/bigstream
ENV ITKWASM_CACHE_DIR=/tmp
ENV PIP_ROOT_USER_ACTION=ignore

# Use the base environment from the baseImage and the conda-env
# from current dir
COPY conda-env.yaml .
RUN mamba env update -n base -f conda-env.yaml

# install bigstream
COPY bigstream bigstream
COPY configs configs

COPY *.py .
COPY *.toml .
COPY *.md .

# Install the Elastix-enabled SimpleITK wheel built in simpleitk-builder.
COPY --from=simpleitk-builder /wheels/*.whl /tmp/simpleitk-wheels/
RUN pip install --no-deps /tmp/simpleitk-wheels/*.whl && \
    rm -rf /tmp/simpleitk-wheels
RUN python -c "import SimpleITK as sitk; assert hasattr(sitk, 'ElastixImageFilter'), 'SimpleITK built without Elastix support'"

RUN pip install -e . && \
    pip install "zarr-tools @ git+https://github.com/JaneliaSciComp/zarr-tools.git@525e55a"
