FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

LABEL org.opencontainers.image.authors="max.michet@robotics.utias.utoronto.ca"

ENV TZ Canada/Eastern
ENV DEBIAN_FRONTEND noninteractive
ENV BLENDER_VERSION 3.6.5

SHELL ["/bin/bash", "-c"]

ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

#? System Packages
COPY packages.txt packages.txt
RUN apt-get update && \
    xargs -a packages.txt apt-get install -y --no-install-recommends && \
    rm packages.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip;

#? Install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install git-lfs && \
    git lfs install

#? Install blender
COPY blender-${BLENDER_VERSION}-linux-x64.tar.xz blender-${BLENDER_VERSION}-linux-x64.tar.xz
RUN tar -xf blender-${BLENDER_VERSION}-linux-x64.tar.xz && \
    rm blender-${BLENDER_VERSION}-linux-x64.tar.xz
# echo "export PATH=/blender-${BLENDER_VERSION}-linux-x64:$PATH" >> ~/.bashrc
# ln -s /usr/local/blender-${BLENDER_VERSION}-linux-x64/blender /usr/local/bin/blender
ENV PATH "/blender-${BLENDER_VERSION}-linux-x64:$PATH"

#? Install blender using apt to install dependencies
RUN apt update && \
    apt install -y blender && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

#? Install basic python packages
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt && \
    rm requirements.txt

#? Install blender python packages
COPY requirements_bpy.txt requirements_bpy.txt
RUN /blender-${BLENDER_VERSION}-linux-x64/3.6/python/bin/python3.10 -m pip install -r requirements_bpy.txt && \
    rm requirements_bpy.txt

#? Refresh CA certificates
RUN update-ca-certificates --fresh
ENV SSL_CERT_DIR=/etc/ssl/certs

#? Create the user
RUN echo "Username is $USERNAME"
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    update-ca-certificates && \
    apt-get install -y --no-install-recommends sudo && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME
    
USER $USERNAME
RUN echo "sudo update-ca-certificates --fresh" >> ~/.bashrc && \
    echo "export SSL_CERT_DIR=/etc/ssl/certs" >> ~/.bashrc
WORKDIR /home/$USERNAME