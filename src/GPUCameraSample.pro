QT += core gui widgets
#opengl
# OTHER LIB PATH
win32: OTHER_LIB_PATH = $$dirname(PWD)/OtherLibs
unix:  OTHER_LIB_PATH = $$dirname(PWD)/OtherLibsLinux
#
include(common_defs.pri)
include(common_funcs.pri)
win32: include(common.pri)
unix:  include(common_unix.pri)

TARGET = $$PROJECT_NAME
TEMPLATE = app

INCLUDEPATH += ./CUDASupport
INCLUDEPATH += ./Camera
INCLUDEPATH += ./Widgets
INCLUDEPATH += $$OTHER_LIB_PATH/FastvideoSDK/core_samples

SOURCES += main.cpp\
        MainWindow.cpp \
    Widgets/GLImageViewer.cpp \
    Globals.cpp \
    AppSettings.cpp \
    CUDASupport/CUDAProcessorBase.cpp \
    FFCReader.cpp \
    FPNReader.cpp \
    ppm.cpp \
    helper_jpeg_load.cpp \
    helper_jpeg_store.cpp \
    Widgets/DenoiseController.cpp \
    Camera/CameraBase.cpp \
    Camera/FrameBuffer.cpp \
    Camera/PGMCamera.cpp \
    RawProcessor.cpp \
    AsyncFileWriter.cpp \
    $$OTHER_LIB_PATH/FastvideoSDK/common/SurfaceTraits.cpp \
    $$OTHER_LIB_PATH/FastvideoSDK/common/alignment.cpp \
    $$OTHER_LIB_PATH/FastvideoSDK/core_samples/SurfaceTraitsInternal.cpp \
    CUDASupport/CUDAProcessorGray.cpp \
    MJPEGEncoder.cpp \
    Camera/GeniCamCamera.cpp \
    rc_genicam_api/buffer.cc \
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

contains( DEFINES, SUPPORT_XIMEA ){
   SOURCES += Camera/XimeaCamera.cpp
}


unix:  SOURCES += rc_genicam_api/gentl_wrapper_linux.cc
win32: SOURCES += rc_genicam_api/gentl_wrapper_win32.cc

HEADERS  += MainWindow.h \
    Widgets/GLImageViewer.h \
    CUDASupport/CUDAProcessorOptions.h \
    Globals.h \
    AppSettings.h \
    CUDASupport/CUDAProcessorBase.h \
    FFCReader.h \
    FPNReader.h \
    ppm.h \
    helper_jpeg.hpp \
    Widgets/DenoiseController.h \
    Camera/CameraBase.h \
    Camera/XimeaCamera.h \
    Camera/FrameBuffer.h \
    Camera/PGMCamera.h \
    RawProcessor.h \
    AsyncFileWriter.h \
    AsyncQueue.h \
    CUDASupport/CUDAProcessorGray.h \
    MJPEGEncoder.h \
    Camera/GeniCamCamera.h \
    rc_genicam_api/buffer.h \
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
    version.h

FORMS    += MainWindow.ui \
    Widgets/DenoiseController.ui

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
    copyPluginsToDestdir(xcbglintegrations)
    copyQtIcuDllsToDestdir($$QT_ICU_DLLS)
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

DISTFILES += \
    res/camera.svg
