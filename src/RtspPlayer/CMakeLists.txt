cmake_minimum_required(VERSION 3.14)

project(RtpPlayer VERSION 1.0 LANGUAGES CXX CUDA)

include_directories(${FastVideo_INC}
                    ${FastVideo_INC}/../../common
                    ${FastVideo_INC}/../../core_samples
                    ${FFMPEG_INC}
                    ${JPEG_INC}
                    ${VideoCodec_INC}
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                    cuviddecoder
                    Widgets
                    .
                )


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
    cuviddecoder/cuviddecoder.cpp
    cuviddecoder/cuviddecoder.h
    Widgets/GLImageViewer.cpp
    Widgets/GLImageViewer.h
    Widgets/GtGWidget.cpp
    Widgets/GtGWidget.h
)

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

message("---> ${VideoCodec_LIB}")

target_link_libraries(RtpPlayer PRIVATE
    Qt5::Core Qt5::Widgets Qt5::Gui Qt5::OpenGL Qt5::Network
    ${FastVideo_LIB}
    ${FFMPEG_LIB}
    ${TURBOJPEG_LIB}
    ${VideoCodec_LIB}
    #${VideoCodec2_LIB}
    CUDA::cudart
    ${ADDITIONAL_LIBS}
)
