#!/bin/bash

BIN_DIR=$1
cd $BIN_DIR
ldconfig -n .
if [ -f "libfastvideo_denoise.so" ]; then
     rm libfastvideo_denoise.so
fi
ln -s libfastvideo_denoise.so.2 libfastvideo_denoise.so

if [ -f "libfastvideo_sdk.so" ]; then
     rm libfastvideo_sdk.so
fi
ln -s libfastvideo_sdk.so.18 libfastvideo_sdk.so
