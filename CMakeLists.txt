cmake_minimum_required(VERSION 3.14)

project(gpu-camera-sample VERSION 1.0 LANGUAGES CUDA CXX)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(CUDAToolkit REQUIRED)

find_package(Qt5 COMPONENTS Gui Widgets OpenGL Network REQUIRED)

find_path(FastVideo_INC
    NAMES fastvideo_sdk.h
    PATHS OtherLibs/fastvideoSDK/fastvideo_sdk/inc/
    REQUIRED)
find_library(FastVideo_LIB
    NAMES fastvideo_sdk
    PATHS OtherLibs/fastvideoSDK/fastvideo_sdk/lib/x64
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
find_library(JPEG_LIB
    NAMES jpeg-static
    PATHS OtherLibs/jpeg-turbo/lib/
    REQUIRED)
find_library(TURBOJPEG_LIB
    NAMES turbojpeg-static
    PATHS OtherLibs/jpeg-turbo/lib/
    REQUIRED)

find_Path(VideoCodec_INC
    NAMES cuviddec.h
    PATHS OtherLibs/nvcodecs/include/
    REQUIRED)
find_library(VideoCodec_LIB
    NAMES nvcuvid
    PATHS OtherLibs/nvcodecs/Lib/x64/
    REQUIRED)
#find_library(VideoCodec2_LIB
#    NAMES nvencodeapi
#    PATHS OtherLibs/nvcodecs/Lib/Win32/
#    REQUIRED)

set(GL_LIBS opengl32)

add_subdirectory(src/CameraSample)
add_subdirectory(src/RtspPlayer)
