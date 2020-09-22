# contains(TARGET_ARCH, arm64 ): QMAKE_CXXFLAGS += -msse3 -m64
OTHER_LIB_PATH = $$dirname(PWD)/OtherLibsLinux


# CUDA
#
contains(TARGET_ARCH, arm64 ) {
    CUDA_TOOLKIT_PATH = "/usr/local/cuda-10.2"
} else {
    CUDA_TOOLKIT_PATH = "/usr/local/cuda-10.2"
}

INCLUDEPATH += $${CUDA_TOOLKIT_PATH}/include
CUDA_LIB  = -L$${CUDA_TOOLKIT_PATH}/lib64
CUDA_LIB += -lnppicc
CUDA_LIB += -lnppig
CUDA_LIB += -lnppif
CUDA_LIB += -lnpps
CUDA_LIB += -lnppc
CUDA_LIB += -lcudart
#
FFMPEG_PATH = $$OTHER_LIB_PATH/ffmpeg
FFMPEG_LIB_PATH  = $$FFMPEG_PATH/lib/

contains( DEFINES, SUPPORT_XIMEA ){
    XI_API_PATH = /opt/XIMEA/
    INCLUDEPATH += $$XI_API_PATH/include
    LIBS += -lm3api
}

contains( DEFINES, SUPPORT_GENICAM ){
    DEFINES -= UNICODE
    DEFINES += GENICAM_NO_AUTO_IMPLIB

    #Ximea Transport Layer
    DEFINES += GENTL_INSTALL_PATH=\'\"/opt/XIMEA/lib\"\'

    #Daheng transport layer
    #DEFINES += GENTL_INSTALL_PATH=\'\"/usr/lib\"\'

    GANAPIPATH = $$OTHER_LIB_PATH/GenICam/library/CPP

contains(TARGET_ARCH, arm64 ) {
    GANAPI_LIB_PATH = $$GANAPIPATH/bin/Linux64_ARM
    GCC_VER = gcc49
}
else {
    GANAPI_LIB_PATH = $$GANAPIPATH/bin/Linux64_x64
    GCC_VER = gcc48
}

    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libGCBase_$${GCC_VER}_v3_2.so
    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libGenApi_$${GCC_VER}_v3_2.so
    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libLog_$${GCC_VER}_v3_2.so
    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libMathParser_$${GCC_VER}_v3_2.so
    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libNodeMapData_$${GCC_VER}_v3_2.so
    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libXmlParser_$${GCC_VER}_v3_2.so

#    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libCLProtocol_$${GCC_VER}_v3_2.so
#    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libFirmwareUpdate_$${GCC_VER}_v3_2.so
#    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/liblog4cpp_$${GCC_VER}_v3_2.so

    INCLUDEPATH += $$GANAPIPATH/include
    LIBS += -L$$GANAPI_LIB_PATH -lCLAllSerial_$${GCC_VER}_v3_2 -lGCBase_$${GCC_VER}_v3_2 -lGenApi_$${GCC_VER}_v3_2
    LIBS += -lLog_$${GCC_VER}_v3_2 -lMathParser_$${GCC_VER}_v3_2 -lNodeMapData_$${GCC_VER}_v3_2 -lXmlParser_$${GCC_VER}_v3_2
#    LIBS += -lFirmwareUpdate_$${GCC_VER}_v3_2 -llog4cpp_$${GCC_VER}_v3_2 -lCLProtocol_$${GCC_VER}_v3_2
}

#
# FASTVIDEO SDK
#
FASTVIDEOPATH  = $$OTHER_LIB_PATH/FastvideoSDK
FASTVIDEO_SDK  = $$FASTVIDEOPATH/fastvideo_sdk
FASTVIDEO_INC  = $$FASTVIDEO_SDK/inc
FASTVIDEO_INC += $$FASTVIDEOPATH/common
FASTVIDEO_INC += $$FASTVIDEOPATH/libs/OpenGL/inc
#
FASTVIDEO_LIB += -L$$FASTVIDEOPATH/fastvideo_sdk/lib/$$PLATFORM
#For ubuntu x64 need to create the dir fastvideo_sdk/lib/Linux64
FASTVIDEO_LIB += -lfastvideo_sdk -lfastvideo_denoise
#
# -lfastvideo_mjpeg -lfastvideo_denoise -lfastvideo_nppFilter -lfastvideo_nppResize -lfastvideo_nppGeometry
#

FASTVIDEO_EXTRA_DLLS += $$FASTVIDEO_SDK/lib/$$PLATFORM/libfastvideo_sdk.so.16.0.0.016000
FASTVIDEO_EXTRA_DLLS += $$FASTVIDEO_SDK/lib/$$PLATFORM/libfastvideo_denoise.so.1.0.1.016000

# NVIDIA VIDEO CODEC SDK
# https://developer.nvidia.com/nvidia-video-codec-sdk/download

contains(TARGET_ARCH, arm64){
    #to work with ffmpeg on nvidia jetson one need to compile it from source (default not work correctly)
    FFMPEG_PATH = $$OTHER_LIB_PATH/ffmpeg
    FFMPEG_LIB = -L$$FFMPEG_PATH/lib/linux/aarch64
    FFMPEG_LIB += -lavformat -lavcodec -lavutil -lswresample -lm -lz -lx264
    FASTVIDEO_EXTRA_DLLS += $$FFMPEG_PATH/lib/linux/aarch64/libavcodec.so.58.106.100
    FASTVIDEO_EXTRA_DLLS += $$FFMPEG_PATH/lib/linux/aarch64/libavformat.so.58.58.100
    FASTVIDEO_EXTRA_DLLS += $$FFMPEG_PATH/lib/linux/aarch64/libavutil.so.56.59.100
    FASTVIDEO_EXTRA_DLLS += $$FFMPEG_PATH/lib/linux/aarch64/libswresample.so.3.8.100
}
else {
    NVCODECS = $$OTHER_LIB_PATH/nvcodecs
    INCLUDEPATH += $$NVCODECS/include
    LIBS += -L$$NVCODECS/Lib/$$PLATFORM -lnvcuvid -lcuda
    FFMPEG_LIB += -lavformat -lavcodec -lavutil -lswresample -lm -lz -lx264
}

#
INCLUDEPATH += $$FASTVIDEO_INC
INCLUDEPATH += $$FFMPEG_PATH/include
#
LIBS += $$FASTVIDEO_LIB
LIBS += $$CUDA_LIB
LIBS += $$FFMPEG_LIB
LIBS += -ldl
LIBS += -ljpeg

#
contains(TARGET_ARCH, arm64 ): LIBS += -lGL
#
defineTest(copyQtIcuDllsToDestdir) {
    DLLS = $$1
    for(ifile, DLLS) {
            IFILE = $$ifile".1"
            OLINK = $$ifile
            QMAKE_POST_LINK += cd $${DESTDIR}; $$QMAKE_COPY $$[QT_INSTALL_LIBS]/$$IFILE .; ln -sf $$IFILE $$OLINK $$escape_expand(\\n\\t)
    }
    export(QMAKE_POST_LINK)
}
# Copies the plugins to the destination directory
defineTest(copySelectedPluginsToDestdir) {
    ifiles = $$1
    for(ifile, ifiles) {
        dname = $${DESTDIR}/$$dirname(ifile)
        makeDir($$dname)
        QMAKE_POST_LINK += $$QMAKE_COPY $$[QT_INSTALL_PLUGINS]/$$ifile $${DESTDIR}/$$ifile $$escape_expand(\\n\\t)
    }
    export(QMAKE_POST_LINK)
}
