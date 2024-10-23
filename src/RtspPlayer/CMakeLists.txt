cmake_minimum_required(VERSION 3.18)

project(RtpPlayer VERSION 1.0 LANGUAGES CXX CUDA)

include_directories(${ADDITIONAL_INCDIR}
                    cuviddecoder
                    Widgets
                    .)

SET(SRC
    common.cpp
    common.h
    common_utils.h
    CTPTransport.cpp
    CTPTransport.h
    DialogOpenServer.cpp
    DialogOpenServer.h
    DialogOpenServer.ui
    fastvideo_decoder.cpp
    fastvideo_decoder.h
    jpegenc.cpp
    jpegenc.h
    main.cpp
    MainWindow.cpp
    MainWindow.h
    MainWindow.ui
    resorces.qrc
    RTSPServer.cpp
    RTSPServer.h
    SDIConverter.cpp
    SDIConverter.h
    vdecoder.cpp
    vdecoder.h
    Widgets/GLImageViewer.cpp
    Widgets/GLImageViewer.h
    Widgets/GtGWidget.cpp
    Widgets/GtGWidget.h
)

if(NOT ${ARCHITECTURE} STREQUAL "aarch64")
    SET(SRC ${SRC}
        cuviddecoder/cuviddecoder.cpp
        cuviddecoder/cuviddecoder.h)
endif()

SET(SRC ${SRC}
    ../../${FASTLIB_DIR}/common/helper_jpeg/helper_jpeg_load.cpp
    ../../${FASTLIB_DIR}/common/helper_jpeg/helper_jpeg.hpp
    ../../${FASTLIB_DIR}/common/helper_jpeg/helper_jpeg_store.cpp
)

if(WIN32)
    add_executable(RtpPlayer WIN32 ${SRC})
else()
    add_executable(RtpPlayer ${SRC})
endif()

target_link_libraries(RtpPlayer PRIVATE
    Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Widgets Qt${QT_VERSION_MAJOR}::Gui Qt${QT_VERSION_MAJOR}::OpenGL Qt${QT_VERSION_MAJOR}::Network
    ${FastVideo_LIB}
    ${FFMPEG_LIB}
    ${TURBOJPEG_LIB}
    #${VideoCodec2_LIB}
    CUDA::cudart
    ${ADDITIONAL_LIBS}
)
