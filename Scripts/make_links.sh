#!/bin/bash

BIN_DIR=$1
cd $BIN_DIR
#Remove all previously created links
find -type l -exec unlink {} \;

ldconfig -n .
ln -s libfastvideo_denoise.so.2 libfastvideo_denoise.so
ln -s libfastvideo_sdk.so.18 libfastvideo_sdk.so
