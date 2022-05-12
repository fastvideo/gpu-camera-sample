cmake_minimum_required(VERSION 3.14)

if(NOT WIN32)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

project(gpu-camera-sample VERSION 1.0 LANGUAGES CUDA CXX)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(CUDAToolkit REQUIRED)

find_package(Qt5 COMPONENTS Gui Widgets OpenGL Network REQUIRED)

if(WIN32)
    set(LIB_DIR OtherLibs)
    set(FASTLIB_DIR ${LIB_DIR}/fastvideoSDK)
    set(FastVideo_PATH ${FASTLIB_DIR}/fastvideo_sdk)
else()
    set(LIB_DIR OtherLibsLinux)
    set(FASTLIB_DIR ${LIB_DIR}/FastvideoSDK)
    set(FastVideo_PATH ${FASTLIB_DIR}/fastvideo_sdk)
endif()

find_path(FastVideo_INC
    NAMES fastvideo_sdk.h
    PATHS ${FastVideo_PATH}/inc/
    REQUIRED)
find_library(FastVideo_LIB
    NAMES fastvideo_sdk
    PATHS ${FastVideo_PATH}/lib/x64 ${FastVideo_PATH}/lib/
    REQUIRED)

find_path(FFMPEG_INC
    NAMES libavutil/opt.h
    PATHS OtherLibs/ffmpeg/include
    REQUIRED)
find_library(AVCODEC_LIB
    NAMES avcodec
    PATHS OtherLibs/ffmpeg/lib
    REQUIRED)
find_library(AVUTIL_LIB
    NAMES avutil
    PATHS OtherLibs/ffmpeg/lib
    REQUIRED)
find_library(AVFORMAT_LIB
    NAMES avformat
    PATHS OtherLibs/ffmpeg/lib
    REQUIRED)
find_library(AVRESAMPLE_LIB
    NAMES swresample
    PATHS OtherLibs/ffmpeg/lib
    REQUIRED)
set(FFMPEG_LIB ${AVCODEC_LIB} ${AVUTIL_LIB} ${AVFORMAT_LIB} ${AVRESAMPLE_LIB})

find_path(JPEG_INC
    NAMES jpeglib.h
    PATHS OtherLibs/jpeg-turbo/include/
    REQUIRED)
if(WIN32)
    find_library(JPEG_LIB
        NAMES jpeg-static
        PATHS OtherLibs/jpeg-turbo/lib/
        REQUIRED)
    find_library(TURBOJPEG_LIB
        NAMES turbojpeg-static
        PATHS OtherLibs/jpeg-turbo/lib/
        REQUIRED)
    set(ADDITIONAL_LIBS ${CUDAToolkit_LIBRARY_DIR}/cuda.lib opengl32)
else()
    set(JPEG_LIB jpeg)
    set(TURBOJPEG_LIB jpeg)
    set(ADDITIONAL_LIBS cuda GL pthread)
endif()

find_Path(VideoCodec_INC
    NAMES cuviddec.h
    PATHS ${LIB_DIR}/nvcodecs/include/ ${LIB_DIR}/nvcodecs/Interface/
    REQUIRED)
find_library(VideoCodec_LIB
    NAMES nvcuvid
    PATHS ${LIB_DIR}/nvcodecs/Lib/x64/ ${LIB_DIR}/nvcodecs/Lib/linux/stubs/x86_64
    REQUIRED)
#find_library(VideoCodec2_LIB
#    NAMES nvencodeapi
#    PATHS OtherLibs/nvcodecs/Lib/Win32/
#    REQUIRED)

add_subdirectory(src/CameraSample)
add_subdirectory(src/RtspPlayer)
