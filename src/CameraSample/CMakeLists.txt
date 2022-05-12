cmake_minimum_required(VERSION 3.18)

project(CameraSample VERSION 1.0 LANGUAGES CUDA)

include_directories(${ADDITIONAL_INCDIR}
                    ${FastVideo_INC}
                    ${FastVideo_INC}/../../common
                    ${FastVideo_INC}/../../core_samples
                    ${FFMPEG_INC}
                    ${JPEG_INC}
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                    avfilewriter
                    Camera
                    CUDASupport
                    RtspServer
                    Widgets
                    .
                )

set(SRC
    AppSettings.cpp
    AsyncFileWriter.cpp
    FFCReader.cpp
    FPNReader.cpp
    Globals.cpp
    helper_jpeg_load.cpp
    helper_jpeg_store.cpp
    main.cpp
    MainWindow.cpp
    MJPEGEncoder.cpp
    ppm.cpp
    RawProcessor.cpp
    quadFragment.frag
    AppSettings.h
    AsyncFileWriter.h
    AsyncQueue.h
    FFCReader.h
    FPNReader.h
    Globals.h
    MainWindow.h
    MJPEGEncoder.h
    ppm.h
    RawProcessor.h
    resource.h
    version.h
    helper_jpeg.hpp
    Resorces.qrc
    #gpu-camera-sample.rc
    resource.rc
    MainWindow.ui
    quadVertex.vert
    avfilewriter/avfilewriter.cpp
    avfilewriter/avfilewriter.h
    Camera/FrameBuffer.cpp
    Camera/FrameBuffer.h
    Camera/GPUCameraBase.cpp
    Camera/GPUCameraBase.h
    Camera/PGMCamera.cpp
    Camera/PGMCamera.h
    Camera/XimeaCamera.cpp
    Camera/XimeaCamera.h
    CUDASupport/CudaAllocator.h
    CUDASupport/CUDAProcessorBase.cpp
    CUDASupport/CUDAProcessorBase.h
    CUDASupport/CUDAProcessorGray.cpp
    CUDASupport/CUDAProcessorGray.h
    CUDASupport/CUDAProcessorOptions.h
    CUDASupport/GPUImage.h
    RtspServer/common_utils.h
    RtspServer/CTPTransport.cpp
    RtspServer/CTPTransport.h
    RtspServer/JpegEncoder.cpp
    RtspServer/JpegEncoder.h
    RtspServer/RTSPStreamerServer.cpp
    RtspServer/RTSPStreamerServer.h
    RtspServer/TcpClient.cpp
    RtspServer/TcpClient.h
    RtspServer/vutils.cpp
    RtspServer/vutils.h
    Widgets/CameraSetupWidget.cpp
    Widgets/CameraSetupWidget.h
    Widgets/CameraSetupWidget.ui
    Widgets/camerastatistics.cpp
    Widgets/camerastatistics.h
    Widgets/camerastatistics.ui
    Widgets/DenoiseController.cpp
    Widgets/DenoiseController.h
    Widgets/DenoiseController.ui
    Widgets/GLImageViewer.cpp
    Widgets/GLImageViewer.h
    Widgets/GtGWidget.cpp
    Widgets/GtGWidget.h
)

set(SRC ${SRC}
    ../../${FASTLIB_DIR}/common/alignment.cpp
    ../../${FASTLIB_DIR}/common/alignment.hpp
    ../../${FASTLIB_DIR}/common/BaseAllocator.cpp
    ../../${FASTLIB_DIR}/common/BaseAllocator.h
    ../../${FASTLIB_DIR}/common/FastAllocator.cpp
    ../../${FASTLIB_DIR}/common/FastAllocator.h
    ../../${FASTLIB_DIR}/common/SurfaceTraits.cpp
    ../../${FASTLIB_DIR}/common/SurfaceTraits.hpp
    ../../${FASTLIB_DIR}/core_samples/SurfaceTraitsInternal.cpp
    ../../${FASTLIB_DIR}/core_samples/SurfaceTraitsInternal.hpp
)

if(WIN32)
    add_executable(CameraSample WIN32 ${SRC})
else()
    add_executable(CameraSample ${SRC})
endif()

target_link_libraries(CameraSample PRIVATE
    Qt5::Core Qt5::Widgets Qt5::Gui Qt5::OpenGL Qt5::Network
    ${FastVideo_LIB}
    ${FFMPEG_LIB}
    ${JPEG_LIB}
    CUDA::cudart
    ${ADDITIONAL_LIBS}
)

if(WIN32)
    get_filename_component(QTBINPATH ${QT_QMAKE_EXECUTABLE} DIRECTORY)
    add_custom_command(
        TARGET CameraSample POST_BUILD
        COMMAND ${QTBINPATH}/windeployqt ${CMAKE_BINARY_DIR}/bin/CameraSample.exe
    )
endif()

#add_custom_command(
#    TARGET CameraSample POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        $<TARGET_FILE:Qt5::Widgets> ${CMAKE_BINARY_DIR}/bin/
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        $<TARGET_FILE:Qt5::Core> ${CMAKE_BINARY_DIR}/bin/
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        $<TARGET_FILE:Qt5::Gui> ${CMAKE_BINARY_DIR}/bin/
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        $<TARGET_FILE:Qt5::Network> ${CMAKE_BINARY_DIR}/bin/
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        $<TARGET_FILE:Qt5::OpenGL> ${CMAKE_BINARY_DIR}/bin/
#)
