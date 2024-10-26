# contains(TARGET_ARCH, arm64 ): QMAKE_CXXFLAGS += -msse3 -m64
OTHER_LIB_PATH = $$dirname(PWD)/OtherLibsLinux
FLIR_PATH = /opt/spinnaker
IMPERX_PATH = $$OTHER_LIB_PATH/Imperx

# CUDA
contains(TARGET_ARCH, arm64){
    #For Jetson Tx2, NX
    CUDA_TOOLKIT_PATH = "/usr/local/cuda-10.2"

    #For Jetson Xavier, Orin
    #CUDA_TOOLKIT_PATH = "/usr/local/cuda-11.4"
    message("target arm64")
}
else {
    CUDA_TOOLKIT_PATH = "/usr/local/cuda-12"
    message("target x86")
}
INCLUDEPATH += $${CUDA_TOOLKIT_PATH}/include
CUDA_LIB  = -L$${CUDA_TOOLKIT_PATH}/lib64
CUDA_LIB += -L$${CUDA_TOOLKIT_PATH}/targets/x86_64-linux/lib/stubs # the place of libcuda on ubuntu 22.04(x86 WSL)
CUDA_LIB += -lnppicc
CUDA_LIB += -lnppig
CUDA_LIB += -lnppif
CUDA_LIB += -lnpps
CUDA_LIB += -lnppc
CUDA_LIB += -lcudart
#
FFMPEG_PATH = $$OTHER_LIB_PATH/ffmpeg
FFMPEG_LIB_PATH  = $$FFMPEG_PATH/lib/

#
# FASTVIDEO SDK
#
FASTVIDEOPATH  = $$OTHER_LIB_PATH/FastvideoSDK
FASTVIDEO_SDK  = $$FASTVIDEOPATH/fastvideo_sdk
FASTVIDEO_INC  = $$FASTVIDEO_SDK/inc
FASTVIDEO_INC += $$FASTVIDEOPATH
FASTVIDEO_INC += $$FASTVIDEOPATH/common
FASTVIDEO_INC += $$FASTVIDEOPATH/libs/OpenGL/inc
#
FASTVIDEO_LIB += -L$$FASTVIDEOPATH/fastvideo_sdk/lib/$$PLATFORM
#For ubuntu x64 we need to create the dir fastvideo_sdk/lib/Linux64
FASTVIDEO_LIB += -lfastvideo_sdk #-lfastvideo_denoise
#
# -lfastvideo_mjpeg -lfastvideo_denoise -lfastvideo_nppFilter -lfastvideo_nppResize -lfastvideo_nppGeometry
#

FASTVIDEO_EXTRA_DLLS += $$FASTVIDEO_SDK/lib/$$PLATFORM/libfastvideo_sdk.so.0.18.1.0.0180100

# NVIDIA VIDEO CODEC SDK
# https://developer.nvidia.com/nvidia-video-codec-sdk/download

contains(TARGET_ARCH, arm64){
    #to work with ffmpeg on nvidia jetson one need to compile it from source (default not work correctly)
    FASTVIDEO_LIB += -L$$FASTVIDEOPATH/fastvideo_sdk/lib/
    FFMPEG_PATH = $$OTHER_LIB_PATH/ffmpeg
    FFMPEG_LIB = -L$$FFMPEG_PATH/lib/linux/aarch64
    FFMPEG_LIB += -lavformat -lavcodec -lavutil -lswresample -lm -lz -lx264

    INCLUDEPATH += $$OTHER_LIB_PATH/libjpeg-turbo/include
    LIBS += -L/$$OTHER_LIB_PATH/libjpeg-turbo/lib64/aarch64 -ljpeg

    FASTVIDEO_EXTRA_DLLS += $$FFMPEG_PATH/lib/linux/aarch64/libavcodec.so.58.18.100
    FASTVIDEO_EXTRA_DLLS += $$FFMPEG_PATH/lib/linux/aarch64/libavformat.so.58.12.100
    FASTVIDEO_EXTRA_DLLS += $$FFMPEG_PATH/lib/linux/aarch64/libavutil.so.56.14.100
    FASTVIDEO_EXTRA_DLLS += $$FFMPEG_PATH/lib/linux/aarch64/libswresample.so.3.1.100

    FASTVIDEO_EXTRA_DLLS += $$OTHER_LIB_PATH/libjpeg-turbo/lib64/aarch64/libjpeg.so.62.3.0

    LIBS += -lGL
}
else {
    FASTVIDEO_LIB += -L$$FASTVIDEOPATH/fastvideo_sdk/lib/
    NVCODECS = $$OTHER_LIB_PATH/nvcodecs
    INCLUDEPATH += $$NVCODECS/include
    INCLUDEPATH += $$NVCODECS/Interface
    LIBS += -L$$NVCODECS/Lib/$$PLATFORM
    LIBS += -L$$NVCODECS/Lib/$$PLATFORM/stubs/x86_64

    LIBS += -lnvcuvid -lcuda
    FFMPEG_LIB += -lavformat -lavcodec -lavutil -lswresample -lm -lz -lx264
}

FASTVIDEO_EXTRA_DLLS += $$PWD/GPUCameraSample.sh

#
INCLUDEPATH += $$FASTVIDEO_INC
INCLUDEPATH += $$FFMPEG_PATH/include
#
LIBS += $$FASTVIDEO_LIB
LIBS += $$CUDA_LIB
LIBS += $$FFMPEG_LIB
LIBS += -ldl
LIBS += -ljpeg

QMAKE_LFLAGS += "-Wl,-rpath,\\\$$ORIGIN"

contains( DEFINES, SUPPORT_XIMEA ){
    XI_API_PATH = /opt/XIMEA/
    INCLUDEPATH += $$XI_API_PATH/include
    LIBS += -lm3api
}

contains( DEFINES, SUPPORT_FLIR ){
    INCLUDEPATH += $$FLIR_PATH/include
    LIBS += -L$$FLIR_PATH/lib -lSpinnaker
}


contains( DEFINES, SUPPORT_IMPERX ){
    INCLUDEPATH += $$IMPERX_PATH/inc
    LIBS += -L$$IMPERX_PATH/lib/Linux64_x64 -lIpxCameraApi
    FASTVIDEO_EXTRA_DLLS += $$IMPERX_PATH/lib/Linux64_x64/libIpxCameraApi.so
    FASTVIDEO_EXTRA_DLLS += $$IMPERX_PATH/lib/Linux64_x64/libIpxCTI.cti
}

contains( DEFINES, SUPPORT_GENICAM ){
    DEFINES -= UNICODE
    DEFINES += GENICAM_NO_AUTO_IMPLIB

    #Ximea Transport Layer
    DEFINES += GENTL_INSTALL_PATH=\'\"/opt/XIMEA/lib\"\'

    #Daheng transport layer
    #DEFINES += GENTL_INSTALL_PATH=\'\"/usr/lib\"\'

    GENAPIPATH = $$OTHER_LIB_PATH/GenICam/library/CPP

contains(TARGET_ARCH, arm64 ) {
    GANAPI_LIB_PATH = $$OTHER_LIB_PATH/GenICam/bin/Linux64_ARM
}
else {
    GANAPI_LIB_PATH = $$OTHER_LIB_PATH/GenICam/bin/Linux64_x64
}
    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libGCBase_$${GENAPIVER}.so
    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libGenApi_$${GENAPIVER}.so
    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libLog_$${GENAPIVER}.so
    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libMathParser_$${GENAPIVER}.so
    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libNodeMapData_$${GENAPIVER}.so
    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libXmlParser_$${GENAPIVER}.so

    INCLUDEPATH += $$GENAPIPATH/include
    LIBS += -L$$GANAPI_LIB_PATH -lGCBase_$${GENAPIVER} -lGenApi_$${GENAPIVER}
    contains(GENAPIVER, v3_0){
    }
    else{
        LIBS +=  -lCLAllSerial_$${GENAPIVER}
    }
    LIBS += -lLog_$${GENAPIVER} -lMathParser_$${GENAPIVER} -lNodeMapData_$${GENAPIVER} -lXmlParser_$${GENAPIVER}

}

contains(DEFINES, SUPPORT_LUCID){

#    LUCID_ROOT = $$LUCID_DEV_ROOT

contains(TARGET_ARCH, arm64 ) {
    LUCID_ROOT = $$OTHER_LIB_PATH/Arena SDK/ArenaSDK_Linux_ARM64
    LIBS += -L"$${LUCID_ROOT}/GenICam/library/lib/Linux64_ARM"
    LIBS += -L"$${LUCID_ROOT}/lib"
}
else {
    LUCID_ROOT = $$OTHER_LIB_PATH/Arena SDK/ArenaSDK_Linux_x64
    LIBS += -L"$${LUCID_ROOT}/GenICam/library/lib/Linux64_x64"
    LIBS += -L"$${LUCID_ROOT}/lib64"
}

    LUCID_GENICAM = $${LUCID_ROOT}/GenICam

    INCLUDEPATH += "$${LUCID_ROOT}/include/Arena"
    INCLUDEPATH += "$${LUCID_ROOT}/include/GenTL"
    INCLUDEPATH += "$${LUCID_GENICAM}/library/CPP/include"




#CONFIG(debug, debug|release){
#    SUF = "d"
##    LUCID_DLL_PATH = $${LUCID_ROOT}/x64Debug
#}else:CONFIG(release, debug|release){
    SUF = ""
#    LUCID_DLL_PATH = $${LUCID_ROOT}/x64Release
#}

    LIBS += -larena$${SUF}
#    LIBS += -lGCBase_gcc54_v3_3_LUCID
#    LIBS += -lGenApi_gcc54_v3_3_LUCID

    LIBS += -lGCBase_gcc54_v3_3_LUCID \
              -lGenApi_gcc54_v3_3_LUCID \
              -lLog_gcc54_v3_3_LUCID \
              -llog4cpp_gcc54_v3_3_LUCID \
              -lMathParser_gcc54_v3_3_LUCID \
              -lNodeMapData_gcc54_v3_3_LUCID \
              -lXmlParser_gcc54_v3_3_LUCID
#    LIBS += -lGenTL_LUCID$${SUF}_v140
#    LIBS += -lGCBase_MD$${SUF}_VC140_v3_3_LUCID
#    LIBS += -lGenApi_MD$${SUF}_VC140_v3_3_LUCID
#    LIBS += -lLog_MD$${SUF}_VC140_v3_3_LUCID
#    LIBS += -lMathParser_MD$${SUF}_VC140_v3_3_LUCID
#    LIBS += -lNodeMapData_MD$${SUF}_VC140_v3_3_LUCID
#    LIBS += -lXmlParser_MD$${SUF}_VC140_v3_3_LUCID
#    LIBS += -llucidlog$${SUF}_v140
    }

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
