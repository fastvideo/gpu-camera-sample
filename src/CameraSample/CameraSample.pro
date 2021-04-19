QT += core gui widgets network opengl

include(../common_defs.pri)
include(../common_funcs.pri)
win32: include(../common.pri)
unix:  include(../common_unix.pri)

TARGET = $$PROJECT_NAME
TEMPLATE = app

unix:  FASTVIDEO_EXTRA_DLLS += $$PWD/GPUCameraSample.sh

CONFIG += console

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
    Camera/FLIRCamera.cpp \
    Camera/GPUCameraBase.cpp \
        MainWindow.cpp \
    Widgets/GLImageViewer.cpp \
    Globals.cpp \
    AppSettings.cpp \
    CUDASupport/CUDAProcessorBase.cpp \
    FFCReader.cpp \
    FPNReader.cpp \
    avfilewriter/avfilewriter.cpp \
    ppm.cpp \
    helper_jpeg_load.cpp \
    helper_jpeg_store.cpp \
    Widgets/DenoiseController.cpp \
    Camera/FrameBuffer.cpp \
    Camera/PGMCamera.cpp \
    RawProcessor.cpp \
    AsyncFileWriter.cpp \
    $$OTHER_LIB_PATH/FastvideoSDK/common/SurfaceTraits.cpp \
    $$OTHER_LIB_PATH/FastvideoSDK/common/alignment.cpp \
    CUDASupport/CUDAProcessorGray.cpp \
    MJPEGEncoder.cpp \
    Camera/GeniCamCamera.cpp \
    Widgets/GtGWidget.cpp \
    Widgets/CameraSetupWidget.cpp \
    RtspServer/CTPTransport.cpp \
    RtspServer/JpegEncoder.cpp \
    RtspServer/RTSPStreamerServer.cpp \
    RtspServer/TcpClient.cpp \
    RtspServer/vutils.cpp

contains( DEFINES, SUPPORT_GENICAM ){
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
    rc_genicam_api/system.cc
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
    rc_genicam_api/system.h
}
contains( DEFINES, SUPPORT_XIMEA ){
   SOURCES += Camera/XimeaCamera.cpp
}

win32: SOURCES += $$OTHER_LIB_PATH/FastvideoSDK/core_samples/SurfaceTraitsInternal.cpp

HEADERS  += MainWindow.h \
    Camera/FLIRCamera.h \
    Camera/GPUCameraBase.h \
    Widgets/GLImageViewer.h \
    CUDASupport/CUDAProcessorOptions.h \
    Globals.h \
    AppSettings.h \
    CUDASupport/CUDAProcessorBase.h \
    FFCReader.h \
    FPNReader.h \
    avfilewriter/avfilewriter.h \
    ppm.h \
    helper_jpeg.hpp \
    Widgets/DenoiseController.h \
    Camera/XimeaCamera.h \
    Camera/FrameBuffer.h \
    Camera/PGMCamera.h \
    RawProcessor.h \
    AsyncFileWriter.h \
    AsyncQueue.h \
    CUDASupport/CUDAProcessorGray.h \
    MJPEGEncoder.h \
    Camera/GeniCamCamera.h \
    Widgets/GtGWidget.h \
    Widgets/CameraSetupWidget.h \
    RtspServer/common_utils.h \
    RtspServer/CTPTransport.h \
    RtspServer/JpegEncoder.h \
    RtspServer/RTSPStreamerServer.h \
    RtspServer/TcpClient.h \
    RtspServer/vutils.h \
    CUDASupport/CudaAllocator.h \
    CUDASupport/GPUImage.h \
    version.h

FORMS    += MainWindow.ui \
    Widgets/DenoiseController.ui \
    Widgets/CameraSetupWidget.ui

RC_FILE = gpu-camera-sample.rc
#resource.rc

unix:copySelectedPluginsToDestdir($$QT_SELECTED_PLUGIN)

#copyPluginsToDestdir(audio)
#win32:copyPluginsToDestdir(sqldrivers)
#copyPluginsToDestdir(printsupport)
win32:copyPluginsToDestdir(platforms)
win32:copyPluginsToDestdir(imageformats)

copyQtDllsToDestdir($$QT_DLLS)
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
copyToDestdir($$FASTVIDEO_EXTRA_DLLS)


RESOURCES += \
    Resorces.qrc

DISTFILES +=
