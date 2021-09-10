#!/bin/bash

mkdir ~/ffmpeg_sources
cd ~/ffmpeg_sources

#install required things from apt
echo "Installing prerequisites"
sudo apt-get update
sudo apt-get -y install autoconf automake build-essential nasm libass-dev libfreetype6-dev libgpac-dev \
  libtool libva-dev libvdpau-dev libxcb1-dev libxcb-shm0-dev \
  libxcb-xfixes0-dev pkg-config texi2html zlib1g-dev libx264-dev libx265-dev

#Install nvidia SDK
echo "Installing the nVidia NVENC SDK."
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
make
sudo make install

#Compile ffmpeg
echo "Compiling ffmpeg"
cd ~/ffmpeg_sources
git clone https://github.com/FFmpeg/FFmpeg -b n4.3.2

cd FFmpeg
./configure \
  --prefix="$HOME/ffmpeg_build" \
  --enable-shared \
  --disable-static \
  --disable-doc \
  --disable-ffplay \
  --disable-ffprobe \
  --enable-cuda-nvcc \
  --enable-cuvid \
  --enable-vaapi \
  --enable-libnpp \
  --extra-cflags="-I/usr/local/cuda/include/" \
  --extra-ldflags=-L/usr/local/cuda/lib64/ \
  --enable-gpl \
  --enable-libx264 \
  --enable-libx265 \
  --enable-nonfree \
  --enable-nvenc

make -j$(nproc)
make -j$(nproc) install
make -j$(nproc) distclean
hash -r

echo "Complete!"
