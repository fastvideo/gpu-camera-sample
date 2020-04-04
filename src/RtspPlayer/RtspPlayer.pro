CONFIG += qt #console
QT += core gui widgets network opengl

include(../common_defs.pri)
include(../common_funcs.pri)
win32: include(../common.pri)
unix:  include(../common_unix.pri)

TARGET = RtspPlayer
TEMPLATE = app

SOURCES = main.cpp \
    $$OTHER_LIB_PATH/FastvideoSDK/common/helper_jpeg/helper_jpeg_load.cpp \
    $$OTHER_LIB_PATH/FastvideoSDK/common/helper_jpeg/helper_jpeg_store.cpp \
    SDIConverter.cpp \
    Widgets/GLImageViewer.cpp \
    Widgets/GtGWidget.cpp \
    CTPTransport.cpp \
    DialogOpenServer.cpp \
    common.cpp \
    fastvideo_decoder.cpp \
    jpegenc.cpp \
    MainWindow.cpp \
    RTSPServer.cpp \
    vdecoder.cpp

FORMS += \
    DialogOpenServer.ui \
    MainWindow.ui

HEADERS += \
    $$OTHER_LIB_PATH/FastvideoSDK/common/helper_jpeg/helper_jpeg.hpp \
    SDIConverter.h \
    Widgets/GLImageViewer.h \
    Widgets/GtGWidget.h \
    common.h \
    CTPTransport.h \
    DialogOpenServer.h \
    fastvideo_decoder.h \
    jpegenc.h \
    MainWindow.h \
    RTSPServer.h \
    common_utils.h \
    vdecoder.h

FFMPEGDIR = $$OTHER_LIB_PATH/ffmpeg

JPEGTURBO = $$OTHER_LIB_PATH/jpeg-turbo

FASTVIDEO = $$OTHER_LIB_PATH/fastvideoSDK

INCLUDEPATH += $$FFMPEGDIR/include \
                $$JPEGTURBO/include \
                $$FASTVIDEO/fastvideo_sdk/inc \
                $$FASTVIDEO/common \
                $$PWD/Widgets

LIBS += -L$$FFMPEGDIR/bin \
        -lavformat -lavcodec -lavutil

win32{
    LIBS += -L$$JPEGTURBO/bin -L$$JPEGTURBO/lib \
            -ljpeg-static -lturbojpeg-static \
            -L$$FASTVIDEO/fastvideo_sdk/bin/x64 -L$$FASTVIDEO/fastvideo_sdk/lib/x64 \
            -lfastvideo_sdk -lfastvideo_mjpeg
}else{
    LIBS += -ljpeg

    QMAKE_CXXFLAGS += -fopenmp
}

win32: QMAKE_CXXFLAGS += /openmp

RESOURCES += \
    resorces.qrc

!contains(TARGET_ARCH, arm64 ){
    include(cuviddecoder/cuviddecoder.pri)
}
