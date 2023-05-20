FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

ARG USER_ID
ARG GROUP_ID

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    zip \
    unzip \
    wget \
    python3-opencv \
    tree \
    vim \
    dvipng texlive-latex-extra texlive-fonts-recommended cm-super \
    libturbojpeg

RUN ln -sf /bin/bash /bin/sh

COPY requirements.txt ./

RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

WORKDIR /home/project
