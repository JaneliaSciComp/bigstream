# Create final image
FROM ghcr.io/janeliascicomp/dask:2025.11.0-py12-ol9

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

RUN pip install -e . && \
    pip install "zarr-tools @ git+https://github.com/JaneliaSciComp/zarr-tools.git@16271a6"
