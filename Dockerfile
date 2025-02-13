# Create final image
FROM ghcr.io/janeliascicomp/dask:2025.1.0-py12-ol9
ARG TARGETPLATFORM

RUN dnf install -y \
        git \
        mesa-libGL

WORKDIR /app/bigstream

ENV ITK_THREADS=
ENV MKL_NUM_THREADS=
ENV NUM_MKL_THREADS=
ENV OPENBLAS_NUM_THREADS=
ENV OPENMP_NUM_THREADS=
ENV OMP_NUM_THREADS=

ENV PYTHONPATH=/app/bigstream

# Use the base environment from the baseImage and the conda-env
# from current dir
COPY conda-env.yaml .
RUN mamba env update -n base -f conda-env.yaml

# install bigstream
COPY scripts scripts
COPY bigstream bigstream
COPY *.py .
COPY *.md .

RUN pip install .
