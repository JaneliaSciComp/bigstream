# Create final image
FROM ghcr.io/janeliascicomp/dask:2023.10.1-py11-ol9
ARG TARGETPLATFORM

RUN dnf install -y \
        git \
        mesa-libGL

WORKDIR /app/bigstream

# install bigstream
COPY conda-env.yaml .
COPY scripts scripts
COPY bigstream bigstream
COPY *.py .
COPY *.md .

# Use the base environment from the baseImage
RUN mamba env update -n base -f conda-env.yaml

RUN pip install .
