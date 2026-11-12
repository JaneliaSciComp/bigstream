# pixi environment baked into the image: `default`, or `fireants` for the
# FireANTs step (adds torch + fireants), e.g. --build-arg PIXI_ENV=fireants
ARG PIXI_ENV=default
ARG GIT_HASH=43d43d4

FROM ghcr.io/janeliascicomp/dask:2026.6.0-py12-ol9 AS builder
ARG PIXI_ENV
ARG GIT_HASH

# git is needed for the git-pinned PyPI deps (fishspot, zarr-tools)
RUN dnf install -y git

ARG PIXI_VERSION=v0.84.1
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=${PIXI_VERSION} PIXI_HOME=/opt/pixi bash
ENV PATH=/opt/pixi/bin:${PATH}

WORKDIR /app/bigstream
# install the locked pixi environment
COPY pyproject.toml pixi.lock setup.py ./
COPY bigstream bigstream
RUN pixi install --locked -e ${PIXI_ENV}

FROM ghcr.io/janeliascicomp/dask:2026.6.0-py12-ol9
ARG PIXI_ENV
ARG GIT_HASH

RUN dnf install -y \
        mesa-libGL

WORKDIR /app/bigstream

ENV ITK_THREADS=
ENV MKL_NUM_THREADS=
ENV NUM_MKL_THREADS=
ENV OPENBLAS_NUM_THREADS=
ENV OPENMP_NUM_THREADS=
ENV OMP_NUM_THREADS=

ENV PYTHONPATH=/app/bigstream
ENV ITKWASM_CACHE_DIR=/tmp

# Same path as in the builder: conda envs hardcode their prefix, and the
# editable bigstream install points at /app/bigstream.
# The pixi env goes first on PATH so python and dask resolve to it
# rather than to the base image's conda env.
COPY --from=builder /app/bigstream/.pixi/envs/${PIXI_ENV} /app/bigstream/.pixi/envs/${PIXI_ENV}
ENV PATH=/app/bigstream/.pixi/envs/${PIXI_ENV}/bin:${PATH}

COPY bigstream bigstream
COPY configs configs

COPY *.py .
COPY *.toml .
COPY *.md .

RUN python -c "import SimpleITK as sitk; print('SITK with Elastix support' if hasattr(sitk, 'ElastixImageFilter') else 'SITK without Elastix support')"
RUN if [ "${PIXI_ENV}" = "fireants" ]; then python -c "import torch, fireants; print(f'torch {torch.__version__}, CUDA build {torch.version.cuda}')"; fi

RUN echo "${GIT_HASH}" > /app/bigstream/.commit-hash
