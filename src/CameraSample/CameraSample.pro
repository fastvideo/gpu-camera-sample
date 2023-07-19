QT += core gui widgets network opengl

include(../common_defs.pri)
include(../common_funcs.pri)
win32: include(../common.pri)
unix:  include(../common_unix.pri)

TARGET = $$PROJECT_NAME
TEMPLATE = app

unix:  FASTVIDEO_EXTRA_DLLS += $$PWD/GPUCameraSample.sh

#CONFIG += console

#INCLUDEPATH += ./CUDASupport
#INCLUDEPATH += ./Camera
#INCLUDEPATH += ./Widgets
INCLUDEPATH += $$OTHER_LIB_PATH/FastvideoSDK/core_samples
INCLUDEPATH += $$PWD
INCLUDEPATH += $$PWD/CUDASupport
INCLUDEPATH += $$PWD/Widgets
INCLUDEPATH += $$PWD/Camera
INCLUDEPATH += $$PWD/RtspServer

SOURCES += main.cpp\
    Camera/BaslerCamera.cpp \
    MainWindow.cpp \
    Globals.cpp \
    AppSettings.cpp \
    FFCReader.cpp \
    FPNReader.cpp \
    ppm.cpp \
    helper_jpeg_load.cpp \
    helper_jpeg_store.cpp \
    RawProcessor.cpp \
    AsyncFileWriter.cpp \
    MJPEGEncoder.cpp \
    avfilewriter/avfilewriter.cpp \
    $$OTHER_LIB_PATH/FastvideoSDK/common/SurfaceTraits.cpp \
    $$OTHER_LIB_PATH/FastvideoSDK/common/alignment.cpp \
    Camera/GPUCameraBase.cpp \
    Camera/FrameBuffer.cpp \
    Camera/PGMCamera.cpp \
    CUDASupport/CUDAProcessorBase.cpp \
    CUDASupport/CUDAProcessorGray.cpp \
    Widgets/DenoiseController.cpp \
    Widgets/GLImageViewer.cpp \
    Widgets/GtGWidget.cpp \
    Widgets/CameraSetupWidget.cpp \
    Widgets/camerastatistics.cpp \
    RtspServer/CTPTransport.cpp \
    RtspServer/JpegEncoder.cpp \
    RtspServer/RTSPStreamerServer.cpp \
    RtspServer/TcpClient.cpp \
    RtspServer/vutils.cpp


win32: SOURCES += $$OTHER_LIB_PATH/FastvideoSDK/core_samples/SurfaceTraitsInternal.cpp

HEADERS  += MainWindow.h \
    Camera/BaslerCamera.h \
    Globals.h \
    AppSettings.h \
    FFCReader.h \
    FPNReader.h \
    ppm.h \
    helper_jpeg.hpp \
    RawProcessor.h \
    AsyncFileWriter.h \
    AsyncQueue.h \
    MJPEGEncoder.h \
    avfilewriter/avfilewriter.h \
    Camera/GPUCameraBase.h \
    Camera/FrameBuffer.h \
    Camera/PGMCamera.h \
    CUDASupport/CUDAProcessorGray.h \
    CUDASupport/CUDAProcessorBase.h \
    CUDASupport/CUDAProcessorOptions.h \
    CUDASupport/CudaAllocator.h \
    CUDASupport/GPUImage.h \
    Widgets/DenoiseController.h \
    Widgets/GLImageViewer.h \
    Widgets/camerastatistics.h \
    Widgets/GtGWidget.h \
    Widgets/CameraSetupWidget.h \
    RtspServer/common_utils.h \
    RtspServer/CTPTransport.h \
    RtspServer/JpegEncoder.h \
    RtspServer/RTSPStreamerServer.h \
    RtspServer/TcpClient.h \
    RtspServer/vutils.h \
    version.h

contains(DEFINES, SUPPORT_XIMEA ){
   SOURCES += Camera/XimeaCamera.cpp
   HEADERS += Camera/XimeaCamera.h
}

contains(DEFINES, SUPPORT_FLIR ){
   SOURCES += Camera/FLIRCamera.cpp
   HEADERS += Camera/FLIRCamera.h
}

contains(DEFINES, SUPPORT_IMPERX ){
   SOURCES += Camera/ImperxCamera.cpp
   HEADERS += Camera/ImperxCamera.h
}

contains(DEFINES, SUPPORT_GENICAM ){
    SOURCES += rc_genicam_api/buffer.cc \
    rc_genicam_api/config.cc \
    rc_genicam_api/cport.cc \
    rc_genicam_api/device.cc \
    rc_genicam_api/exception.cc \
    rc_genicam_api/image.cc \
    rc_genicam_api/imagelist.cc \
    rc_genicam_api/interface.cc \
    rc_genicam_api/pointcloud.cc \
    rc_genicam_api/stream.cc \
    rc_genicam_api/system.cc \
    Camera/GeniCamCamera.cpp

    unix:  SOURCES += rc_genicam_api/gentl_wrapper_linux.cc
    win32: SOURCES += rc_genicam_api/gentl_wrapper_win32.cc

    HEADERS  +=  rc_genicam_api/buffer.h \
    rc_genicam_api/config.h \
    rc_genicam_api/cport.h \
    rc_genicam_api/device.h \
    rc_genicam_api/exception.h \
    rc_genicam_api/gentl_wrapper.h \
    rc_genicam_api/image.h \
    rc_genicam_api/imagelist.h \
    rc_genicam_api/interface.h \
    rc_genicam_api/pixel_formats.h \
    rc_genicam_api/pointcloud.h \
    rc_genicam_api/stream.h \
    rc_genicam_api/system.h \
    Camera/GeniCamCamera.h
}

contains(DEFINES, SUPPORT_LUCID ){
   SOURCES += Camera/LucidCamera.cpp
   HEADERS += Camera/LucidCamera.h
}

contains(TARGET_ARCH, arm64 ) {
    contains(DEFINES, SUPPORT_MIPI){
        HEADERS += Camera/MIPICamera.h
        SOURCES += Camera/MIPICamera.cpp
    }
}else{
    DEFINES -= SUPPORT_MIPI
}

FORMS    += MainWindow.ui \
    Widgets/DenoiseController.ui \
    Widgets/CameraSetupWidget.ui \
    Widgets/camerastatistics.ui

win32{
    SOURCES += $$PWD/../../OtherLibs/fastvideoSDK/common/BaseAllocator.cpp \
               $$PWD/../../OtherLibs/fastvideoSDK/common/FastAllocator.cpp

    HEADERS += $$PWD/../../OtherLibs/fastvideoSDK/common/BaseAllocator.h \
               $$PWD/../../OtherLibs/fastvideoSDK/common/FastAllocator.h
}else{
    SOURCES += $$PWD/../../OtherLibsLinux/FastvideoSDK/common/BaseAllocator.cpp \
               $$PWD/../../OtherLibsLinux/FastvideoSDK/common/FastAllocator.cpp

    HEADERS += $$PWD/../../OtherLibsLinux/FastvideoSDK/common/BaseAllocator.h \
               $$PWD/../../OtherLibsLinux/FastvideoSDK/common/FastAllocator.h
}

RC_FILE = gpu-camera-sample.rc
#resource.rc

unix:copySelectedPluginsToDestdir($$QT_SELECTED_PLUGIN)

#copyPluginsToDestdir(audio)
#win32:copyPluginsToDestdir(sqldrivers)
#copyPluginsToDestdir(printsupport)
win32:copyPluginsToDestdir(platforms)
win32:copyPluginsToDestdir(imageformats)

copyQtDllsToDestdir($$QT_DLLS)
copyToDestdir($$FASTVIDEO_EXTRA_DLLS)

unix {

    contains(TARGET_ARCH, arm64){
        include($$PWD/jetson_api/jetson_api.pri)
    }

    copyPluginsToDestdir(xcbglintegrations)
    copyQtIcuDllsToDestdir($$QT_ICU_DLLS)
    makeLinks()
}

win32 {
    copyToDestdir($$VC_REDIST_DLLS)
    for(ifile, QT_EXECS) {
        copyToDestdir($$[QT_INSTALL_BINS]/$$ifile)
    }

    copyToDestdir($$FASTVIDEO_DLL)
    copyToDestdir($$CUDA_DLL)
}


RESOURCES += \
    Resorces.qrc

DISTFILES +=
