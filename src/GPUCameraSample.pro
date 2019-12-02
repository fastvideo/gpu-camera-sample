QT += core gui widgets

include(common_defs.pri)
include(common_funcs.pri)
include(common.pri)

TARGET = $$PROJECT_NAME
TEMPLATE = app

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
    Camera/XimeaCamera.cpp \
    Camera/FrameBuffer.cpp \
    Camera/PGMCamera.cpp \
    RawProcessor.cpp \
    AsyncFileWriter.cpp

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
    AsyncQueue.h

FORMS    += MainWindow.ui \
    Widgets/DenoiseController.ui

unix:copySelectedPluginsToDestdir($$QT_SELECTED_PLUGIN)

#copyPluginsToDestdir(audio)
#win32:copyPluginsToDestdir(sqldrivers)
#copyPluginsToDestdir(printsupport)
win32:copyPluginsToDestdir(platforms)
copyPluginsToDestdir(imageformats)

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
    copyToDestdir($$FASTVIDEO_EXTRA_DLLS)
}

RESOURCES += \
    Resorces.qrc
