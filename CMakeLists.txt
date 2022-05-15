cmake_minimum_required(VERSION 3.18)

if(NOT WIN32)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

project(gpu-camera-sample VERSION 1.0 LANGUAGES CUDA CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(WIN32)
    set(ARCHITECTURE WIN32)
else()
    EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
endif()

message("target architecture ${ARCHITECTURE}")

option(USE_XIMEA ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(CUDAToolkit REQUIRED)
find_package(Qt5 COMPONENTS Gui Widgets OpenGL Network REQUIRED)

set(ADDITIONAL_INCDIR)

if(WIN32)
    set(LIB_DIR OtherLibs)
    set(FASTLIB_DIR ${LIB_DIR}/fastvideoSDK)
    set(FastVideo_PATH ${FASTLIB_DIR}/fastvideo_sdk)
else()
    set(LIB_DIR OtherLibsLinux)
    set(FASTLIB_DIR ${LIB_DIR}/FastvideoSDK)
    set(FastVideo_PATH ${FASTLIB_DIR}/fastvideo_sdk)
endif()

if(USE_XIMEA)
    add_definitions(-DSUPPORT_XIMEA)
    set(Ximea_PATH D:/devs/XIMEA/API/xiAPI/)
    find_path(Ximea_INC
        NAMES xiApi.h
        PATHS ${Ximea_PATH}
        REQUIRED)
    find_library(Ximea_LIB
        NAMES xiapi64
        PATHS ${Ximea_PATH}
        REQUIRED)

    if(WIN32)
        set(THIRD_PARTY_DLLS ${THIRD_PARTY_DLLS} ${Ximea_PATH}/xiapi64.dll)
    endif()
    set(ADDITIONAL_INCDIR ${ADDITIONAL_INCDIR} ${Ximea_INC})
endif()

find_path(FastVideo_INC
    NAMES fastvideo_sdk.h
    PATHS ${FastVideo_PATH}/inc/
    REQUIRED)
find_library(FastVideo_LIB
    NAMES fastvideo_sdk
    PATHS ${FastVideo_PATH}/lib/x64 ${FastVideo_PATH}/lib/
    REQUIRED)

set(FFMPEG_PATH OtherLibs/ffmpeg)
find_path(FFMPEG_INC
    NAMES libavutil/opt.h
    PATHS ${FFMPEG_PATH}/include
    REQUIRED)
find_library(AVCODEC_LIB
    NAMES avcodec
    PATHS ${FFMPEG_PATH}/lib
    REQUIRED)
find_library(AVUTIL_LIB
    NAMES avutil
    PATHS ${FFMPEG_PATH}/lib
    REQUIRED)
find_library(AVFORMAT_LIB
    NAMES avformat
    PATHS ${FFMPEG_PATH}/lib
    REQUIRED)
find_library(AVRESAMPLE_LIB
    NAMES swresample
    PATHS ${FFMPEG_PATH}/lib
    REQUIRED)
set(FFMPEG_LIB ${AVCODEC_LIB} ${AVUTIL_LIB} ${AVFORMAT_LIB} ${AVRESAMPLE_LIB})

if(WIN32)
    set(THIRD_PARTY_DLLS ${THIRD_PARTY_DLLS}
        OtherLibs/ffmpeg/bin/avcodec-57.dll
        OtherLibs/ffmpeg/bin/avformat-57.dll
        OtherLibs/ffmpeg/bin/avutil-55.dll
        OtherLibs/ffmpeg/bin/swresample-2.dll
        ${CUDAToolkit_LIBRARY_ROOT}/bin/cudart64_110.dll
    )
endif()

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

set(ADDITIONAL_LIBS
    ${ADDITIONAL_LIBS}
    ${Ximea_LIB}
)

if(NOT ${ARCHITECTURE} STREQUAL "aarch64")
    find_path(VideoCodec_INC
        NAMES cuviddec.h
        PATHS ${LIB_DIR}/nvcodecs/include/ ${LIB_DIR}/nvcodecs/Interface/
        REQUIRED)
    find_library(VideoCodec_LIB
        NAMES nvcuvid
        PATHS ${LIB_DIR}/nvcodecs/Lib/x64/ ${LIB_DIR}/nvcodecs/Lib/linux/stubs/x86_64
        REQUIRED)

    set(ADDITIONAL_INCDIR ${ADDITIONAL_INCDIR}
        ${VideoCodec_INC})

    set(ADDITIONAL_LIBS ${ADDITIONAL_LIBS}
        ${VideoCodec_LIB})
endif()
#find_library(VideoCodec2_LIB
#    NAMES nvencodeapi
#    PATHS OtherLibs/nvcodecs/Lib/Win32/
#    REQUIRED)

set(ADDITIONAL_INCDIR ${ADDITIONAL_INCDIR}
    ${FastVideo_INC}
    ${FastVideo_INC}/../../common
    ${FastVideo_INC}/../../core_samples
    ${FFMPEG_INC}
    ${JPEG_INC}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(src/CameraSample)
add_subdirectory(src/RtspPlayer)

# create a list of files to copy
set(THIRD_PARTY_DLLS ${THIRD_PARTY_DLLS}
    ${FASTLIB_DIR}/fastvideo_sdk/bin/x64/fastvideo_sdk.dll
)

install(FILES ${THIRD_PARTY_DLLS}
    DESTINATION ${CMAKE_BINARY_DIR}/bin/
)
