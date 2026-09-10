FROM mcr.microsoft.com/devcontainers/base:ubuntu24.04

# Install firewall dependencies and pixi
RUN apt-get update && apt-get install -y --no-install-recommends \
    iptables \
    ipset \
    dnsutils \
    jq \
    aggregate \
    curl \
    nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Install conda system-wide
RUN wget https://github.com/conda-forge/miniforge/releases/download/26.3.2-3/Miniforge3-26.3.2-3-Linux-$(uname -m).sh \
    -O mamba-install.sh && \
    bash mamba-install.sh -b -p /opt/mambaforge

ENV PATH=/opt/mambaforge/bin:${PATH}

# Copy firewall init script
COPY init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod +x /usr/local/bin/init-firewall.sh
