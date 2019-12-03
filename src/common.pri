QMAKE_CXXFLAGS += "/openmp"

win32: QMAKE_CXXFLAGS_RELEASE -= -Zc:strictStrings
win32: QMAKE_CFLAGS_RELEASE -= -Zc:strictStrings
win32: QMAKE_CFLAGS -= -Zc:strictStrings
win32: QMAKE_CXXFLAGS -= -Zc:strictStrings

QMAKE_LFLAGS_WINDOWS += "/MACHINE:X64"
QMAKE_LFLAGS_WINDOWS += "/LARGEADDRESSAWARE"

FASTVIDEOPATH = $$PWD/../OtherLibs/fastvideoSDK
FASTVIDEO_SDK = $$FASTVIDEOPATH/fastvideo_sdk

FASTVIDEO_INC  = $$FASTVIDEO_SDK/inc
FASTVIDEO_INC += $$FASTVIDEOPATH/common
FASTVIDEO_INC += $$FASTVIDEOPATH/libs/OpenGL/inc

FFMPEG_PATH = $$FASTVIDEOPATH/libs/ffmpeg
FFMPEG_LIB  = $$FFMPEG_PATH/lib/win/$$PLATFORM

FASTVIDEO_LIB  = -L$$FASTVIDEO_SDK/lib/$$PLATFORM
FASTVIDEO_DLL_PATH = $$FASTVIDEO_SDK/bin/$$PLATFORM

FASTVIDEO_DLL  = $$FASTVIDEO_DLL_PATH/fastvideo_sdk.dll
FASTVIDEO_DLL += $$FASTVIDEO_DLL_PATH/fastvideo_mjpeg.dll

#FASTVIDEO_DLL += $$FASTVIDEO_DLL_PATH/fastvideo_experimentalImageFilter.dll
FASTVIDEO_DLL += $$FASTVIDEO_DLL_PATH/fastvideo_denoise.dll

FASTVIDEO_EXTRA_DLLS += $$FASTVIDEOPATH/libs/senselock/$$PLATFORM/sense4.dll

#FASTVIDEO_DLL += $$FASTVIDEO_DLL_PATH/fastvideo_nppFilter.dll
#FASTVIDEO_DLL += $$FASTVIDEO_DLL_PATH/fastvideo_nppResize.dll
#FASTVIDEO_DLL += $$FASTVIDEO_DLL_PATH/fastvideo_nppGeometry.dll

FASTVIDEO_LIB += -lfastvideo_sdk \
#        -lfastvideo_mjpeg \
        -lfastvideo_denoise
#        -lfastvideo_experimentalImageFilter \
#        -lfastvideo_nppFilter \
#        -lfastvideo_nppResize \
#        -lfastvideo_jpegLosslessDecoder \
#        -lfastvideo_nppGeometry

FASTVIDEO_EXTRA_LIBS += -L$$FASTVIDEOPATH/libs/OpenGL/lib/win/$$PLATFORM
FASTVIDEO_EXTRA_DLLS += $$FFMPEG_LIB/src/ffmpeg-3.4.2.tar.gz
FASTVIDEO_EXTRA_DLLS += $$FFMPEG_LIB/avcodec-57.dll
FASTVIDEO_EXTRA_DLLS += $$FFMPEG_LIB/avformat-57.dll
FASTVIDEO_EXTRA_DLLS += $$FFMPEG_LIB/avutil-55.dll
FASTVIDEO_EXTRA_DLLS += $$FFMPEG_LIB/swresample-2.dll

FASTVIDEO_EXTRA_DLLS += $$[QT_INSTALL_BINS]/icudt58.dll
FASTVIDEO_EXTRA_DLLS += $$[QT_INSTALL_BINS]/icuin58.dll
FASTVIDEO_EXTRA_DLLS += $$[QT_INSTALL_BINS]/icuuc58.dll
#FASTVIDEO_EXTRA_DLLS += $$LIBTIFF_LIB/tiff.dll

CUDA_TOOLKIT_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1"
CUDAINC += $$CUDA_TOOLKIT_PATH/include
CUDA_DLL_PATH = $$CUDA_TOOLKIT_PATH/bin
CUDA_DLL += $$CUDA_DLL_PATH/cudart64_101.dll

CUDA_DLL += $$CUDA_DLL_PATH/nppc64_10.dll
CUDA_DLL += $$CUDA_DLL_PATH/nppif64_10.dll
CUDA_DLL += $$CUDA_DLL_PATH/nppicc64_10.dll
CUDA_DLL += $$CUDA_DLL_PATH/nppig64_10.dll


INCLUDEPATH += $$PWD
INCLUDEPATH += $$PWD/CUDASupport
INCLUDEPATH += $$PWD/Widgets
INCLUDEPATH += $$PWD/Camera
INCLUDEPATH += $$CUDAINC
INCLUDEPATH += $$FASTVIDEO_INC
INCLUDEPATH += $$FASTVIDEOPATH\core_samples
INCLUDEPATH += $$FFMPEG_PATH/inc

QMAKE_CXXFLAGS += "/WX" # Treats all compiler warnings as errors.
QMAKE_LFLAGS_WINDOWS += "/NODEFAULTLIB:LIBCMT"

QT_DLLS += Qt5Core
QT_DLLS += Qt5Gui
QT_DLLS += Qt5Widgets
QT_DLLS += Qt5Svg
QT_DLLS += Qt5Xml
QT_DLLS += Qt5Multimedia
QT_DLLS += Qt5OpenGL
QT_DLLS += Qt5Network

CONFIG(debug, debug|release){
    QMAKE_LFLAGS_WINDOWS += "/NODEFAULTLIB:LIBCMTD"
    QMAKE_LFLAGS_WINDOWS += "/NODEFAULTLIB:MSVCRT"
}else:CONFIG(release, debug|release){
    #For release build debug
    QMAKE_LFLAGS_WINDOWS += "/INCREMENTAL:NO /DEBUG /OPT:REF /OPT:ICF"
    QMAKE_CXXFLAGS += "/Zi /DEBUG"
}

LIBS += $$FASTVIDEO_LIB
LIBS += $$FASTVIDEO_EXTRA_LIBS
LIBS += -L$$FFMPEG_LIB  -lavcodec -lavformat -lavutil -lswresample
LIBS += -L$$CUDA_TOOLKIT_PATH/lib/$$PLATFORM -lcudart -lcuda
LIBS += -lglu32 -lopengl32 -lgdi32 -luser32 -lMscms -lShell32 -lOle32 -lWs2_32 -lstrmiids -lComdlg32

contains( DEFINES, SUPPORT_XIMEA ){
    XI_API_PATH = $$PWD/../OtherLibs/XIMEA/API
    INCLUDEPATH += $$XI_API_PATH
    LIBS += -L$$XI_API_PATH/x64 -lxiapi64
    FASTVIDEO_EXTRA_DLLS += $$XI_API_PATH/x64/xiapi64.dll
}


contains( DEFINES, USE_NV_API ){
    NVAPI_PATH = "C:/Program Files (x86)/NVIDIA Corporation/Nsight Visual Studio Edition 5.6/Monitor/nvapi"
    INCLUDEPATH += $$NVAPI_PATH
    LIBS += -L$$NVAPI_PATH/amd64 -lnvapi64
}

HEADERS += \
    $$PWD/version.h

