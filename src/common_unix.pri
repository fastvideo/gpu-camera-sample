QMAKE_CXXFLAGS += -msse3 -m64
#
# CUDA
#
CUDA_TOOLKIT_PATH = "/usr/local/cuda-10.1"
INCLUDEPATH += $${CUDA_TOOLKIT_PATH}/include
CUDA_LIB  = -L$${CUDA_TOOLKIT_PATH}/lib64
CUDA_LIB += -lnppicc
CUDA_LIB += -lnppig
CUDA_LIB += -lnppif
CUDA_LIB += -lnpps
CUDA_LIB += -lnppc
CUDA_LIB += -lcudart
#
contains( DEFINES, SUPPORT_XIMEA ){
    XI_API_PATH = /opt/XIMEA/
    INCLUDEPATH += $$XI_API_PATH/include
    LIBS += -lm3api
}

contains( DEFINES, SUPPORT_GENICAM ){
    DEFINES -= UNICODE
    DEFINES += GENICAM_NO_AUTO_IMPLIB
    #
    # Setup Ximea Transport Layer !!!!
    #
    DEFINES += GENTL_INSTALL_PATH=\'\"/opt/XIMEA/lib\"\'

    GANAPIPATH = $$OTHER_LIB_PATH/GenICam/library/CPP

    GANAPI_LIB_PATH = $$GANAPIPATH/bin/Linux64_x64

    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libGCBase_gcc48_v3_2.so
    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libGenApi_gcc48_v3_2.so
    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libLog_gcc48_v3_2.so
    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libMathParser_gcc48_v3_2.so
    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libNodeMapData_gcc48_v3_2.so
    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libXmlParser_gcc48_v3_2.so

#    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libCLProtocol_gcc48_v3_2.so
#    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/libFirmwareUpdate_gcc48_v3_2.so
#    FASTVIDEO_EXTRA_DLLS += $$GANAPI_LIB_PATH/liblog4cpp_gcc48_v3_2.so


    INCLUDEPATH += $$GANAPIPATH/include
    LIBS += -L$$GANAPI_LIB_PATH -lCLAllSerial_gcc48_v3_2 -lGCBase_gcc48_v3_2 -lGenApi_gcc48_v3_2
    LIBS += -lLog_gcc48_v3_2 -lMathParser_gcc48_v3_2 -lNodeMapData_gcc48_v3_2 -lXmlParser_gcc48_v3_2
#    LIBS += -lFirmwareUpdate_gcc48_v3_2 -llog4cpp_gcc48_v3_2 -lCLProtocol_gcc48_v3_2
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
FASTVIDEO_LIB += -L$$FASTVIDEOPATH/fastvideo_sdk/lib
FASTVIDEO_LIB += -lfastvideo_sdk -lfastvideo_denoise
#
# -lfastvideo_mjpeg -lfastvideo_denoise -lfastvideo_nppFilter -lfastvideo_nppResize -lfastvideo_nppGeometry
#
FFMPEG_PATH = $$OTHER_LIB_PATH/FastvideoSDK/libs/ffmpeg
FFMPEG_LIB = -L$$FFMPEG_PATH/lib/linux/x86_64/
FFMPEG_LIB += -lavcodec -lavformat -lavutil -lswresample
#
INCLUDEPATH += $$FASTVIDEO_INC
INCLUDEPATH += $$FFMPEG_PATH/inc
#
LIBS += $$FASTVIDEO_LIB
LIBS += $$CUDA_LIB
LIBS += $$FFMPEG_LIB
LIBS += -ldl
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
