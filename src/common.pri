QMAKE_CXXFLAGS += "/openmp"

QMAKE_CXXFLAGS_RELEASE -= -Zc:strictStrings
QMAKE_CFLAGS_RELEASE -= -Zc:strictStrings
QMAKE_CFLAGS -= -Zc:strictStrings
QMAKE_CXXFLAGS -= -Zc:strictStrings

QMAKE_LFLAGS_WINDOWS += "/MACHINE:X64"
QMAKE_LFLAGS_WINDOWS += "/LARGEADDRESSAWARE"

OTHER_LIB_PATH = $$dirname(PWD)/OtherLibs

JPEGTURBO = $$OTHER_LIB_PATH/jpeg-turbo

FASTVIDEOPATH = $$OTHER_LIB_PATH/FastvideoSDK
FASTVIDEO_SDK = $$FASTVIDEOPATH/fastvideo_sdk

FASTVIDEO_INC  = $$FASTVIDEO_SDK/inc
FASTVIDEO_INC += $$FASTVIDEOPATH/common
FASTVIDEO_INC += $$FASTVIDEOPATH/libs/OpenGL/inc

FFMPEG_PATH = $$OTHER_LIB_PATH/ffmpeg
FFMPEG_SRC  = $$FFMPEG_PATH/src/ffmpeg-4.0.2.tar.bz2
FFMPEG_LIB  = $$FFMPEG_PATH/lib

FLIR_PATH = $$OTHER_LIB_PATH/FLIR

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
FASTVIDEO_EXTRA_DLLS += $$FFMPEG_LIB/avcodec-58.dll
FASTVIDEO_EXTRA_DLLS += $$FFMPEG_LIB/avformat-58.dll
FASTVIDEO_EXTRA_DLLS += $$FFMPEG_LIB/avutil-56.dll
FASTVIDEO_EXTRA_DLLS += $$FFMPEG_LIB/swresample-3.dll

FASTVIDEO_EXTRA_DLLS += $$FFMPEG_SRC

#FASTVIDEO_EXTRA_DLLS += $$[QT_INSTALL_BINS]/icudt58.dll
#FASTVIDEO_EXTRA_DLLS += $$[QT_INSTALL_BINS]/icuin58.dll
#FASTVIDEO_EXTRA_DLLS += $$[QT_INSTALL_BINS]/icuuc58.dll
#FASTVIDEO_EXTRA_DLLS += $$LIBTIFF_LIB/tiff.dll

CUDA_TOOLKIT_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2"
CUDAINC += $$CUDA_TOOLKIT_PATH/include
CUDA_DLL_PATH = $$CUDA_TOOLKIT_PATH/bin
CUDA_DLL += $$CUDA_DLL_PATH/cudart64_102.dll

CUDA_DLL += $$CUDA_DLL_PATH/nppc64_10.dll
CUDA_DLL += $$CUDA_DLL_PATH/nppif64_10.dll
CUDA_DLL += $$CUDA_DLL_PATH/nppicc64_10.dll
CUDA_DLL += $$CUDA_DLL_PATH/nppig64_10.dll

# NVIDIA VIDEO CODEC SDK
# https://developer.nvidia.com/nvidia-video-codec-sdk/download
NVCODECS = $$OTHER_LIB_PATH/nvcodecs
INCLUDEPATH += $$NVCODECS/Interface
LIBS += -L$$NVCODECS/Lib/$$PLATFORM -lnvcuvid

INCLUDEPATH += $$CUDAINC
INCLUDEPATH += $$FASTVIDEO_INC
INCLUDEPATH += $$FASTVIDEOPATH/core_samples
INCLUDEPATH += $$FFMPEG_PATH/include
INCLUDEPATH += $$JPEGTURBO/include

QMAKE_CXXFLAGS += "/WX" # Treats all compiler warnings as errors.
#QMAKE_LFLAGS_WINDOWS += "/NODEFAULTLIB:LIBCMT"

QT_DLLS += Qt5Core
QT_DLLS += Qt5Gui
QT_DLLS += Qt5Widgets
QT_DLLS += Qt5Svg
QT_DLLS += Qt5Network
QT_DLLS += Qt5OpenGL
#QT_DLLS += Qt5Xml
#QT_DLLS += Qt5Multimedia


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
LIBS += -L$$JPEGTURBO/lib -ljpeg-static -lturbojpeg-static

contains( DEFINES, SUPPORT_XIMEA ){
    XI_API_PATH = $$OTHER_LIB_PATH/XIMEA/API
    INCLUDEPATH += $$XI_API_PATH
    LIBS += -L$$XI_API_PATH/x64 -lxiapi64
    FASTVIDEO_EXTRA_DLLS += $$XI_API_PATH/x64/xiapi64.dll
}

contains( DEFINES, SUPPORT_FLIR ){
    INCLUDEPATH += $$FLIR_PATH/include
    LIBS += -L$$FLIR_PATH/lib64/vs2015 -lSpinnaker_v140
    FASTVIDEO_EXTRA_DLLS += $$FLIR_PATH/bin64/vs2015/Spinnaker_v140.dll
}

contains( DEFINES, SUPPORT_GENICAM ){
    DEFINES -= UNICODE
    DEFINES += GENICAM_NO_AUTO_IMPLIB

    GANAPIPATH = $$OTHER_LIB_PATH/GenICam/library/CPP

    FASTVIDEO_EXTRA_DLLS += $$GANAPIPATH/bin/Win64_x64/GCBase_MD_VC141_v3_2.dll
    FASTVIDEO_EXTRA_DLLS += $$GANAPIPATH/bin/Win64_x64/GenApi_MD_VC141_v3_2.dll
    FASTVIDEO_EXTRA_DLLS += $$GANAPIPATH/bin/Win64_x64/GenCP_MD_VC141_v3_2.dll
    FASTVIDEO_EXTRA_DLLS += $$GANAPIPATH/bin/Win64_x64/log4cpp_MD_VC141_v3_2.dll
    FASTVIDEO_EXTRA_DLLS += $$GANAPIPATH/bin/Win64_x64/Log_MD_VC141_v3_2.dll
    FASTVIDEO_EXTRA_DLLS += $$GANAPIPATH/bin/Win64_x64/XmlParser_MD_VC141_v3_2.dll
    FASTVIDEO_EXTRA_DLLS += $$GANAPIPATH/bin/Win64_x64/MathParser_MD_VC141_v3_2.dll
    FASTVIDEO_EXTRA_DLLS += $$GANAPIPATH/bin/Win64_x64/NodeMapData_MD_VC141_v3_2.dll

    INCLUDEPATH += $$GANAPIPATH/include
    LIBS += -L$$GANAPIPATH/lib/Win64_x64 -lGCBase_MD_VC141_v3_2 -lGenApi_MD_VC141_v3_2 -lGenCP_MD_VC141_v3_2
    LIBS += -llog4cpp_MD_VC141_v3_2 -lLog_MD_VC141_v3_2 -lXmlParser_MD_VC141_v3_2
}

contains( DEFINES, USE_NV_API ){
    NVAPI_PATH = "C:/Program Files (x86)/NVIDIA Corporation/Nsight Visual Studio Edition 5.6/Monitor/nvapi"
    INCLUDEPATH += $$NVAPI_PATH
    LIBS += -L$$NVAPI_PATH/amd64 -lnvapi64
}
