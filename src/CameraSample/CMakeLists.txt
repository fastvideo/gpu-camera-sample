cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME CameraSample)

project(${PROJECT_NAME} VERSION 1.0 LANGUAGES CUDA)

include_directories(${ADDITIONAL_INCDIR}
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

if(${ARCHITECTURE} STREQUAL "aarch64")
    set(TEGRA_ARMABI aarch64-linux-gnu)

    find_library(NVBUF_LIB
        NAMES nvbuf_utils
        PATHS /usr/lib/$${TEGRA_ARMABI}/ /usr/lib/${TEGRA_ARMABI}/tegra/
        REQUIRED)
    find_library(V4L2_LIB
        NAMES v4l2 REQUIRED)
    list(APPEND ADDITIONAL_LIBS
        ${NVBUF_LIB}
        ${V4L2_LIB})

    include_directories(jetson_api)
    SET(SRC ${SRC}
        jetson_api/common_types.h
        jetson_api/jetson_api.pri
        jetson_api/nvbuf_utils.h
        jetson_api/nvvideoencoder.cpp
        jetson_api/nvvideoencoder.h
        jetson_api/v4l2encoder.cpp
        jetson_api/v4l2encoder.h
        jetson_api/v4l2_nv_extensions.h
        )
endif()

if(WIN32)
    add_executable(${PROJECT_NAME} WIN32 ${SRC})
else()
    add_executable(${PROJECT_NAME} ${SRC})
endif()

if(USE_XIMEA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        -DSUPPORT_XIMEA=1
    )
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Widgets Qt${QT_VERSION_MAJOR}::Gui Qt${QT_VERSION_MAJOR}::OpenGL Qt${QT_VERSION_MAJOR}::Network
    ${FastVideo_LIB}
    ${FFMPEG_LIB}
    ${JPEG_LIB}
    CUDA::cudart
    ${ADDITIONAL_LIBS}
)

if(WIN32)
    get_filename_component(QTBINPATH ${QT_QMAKE_EXECUTABLE} DIRECTORY)
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${QTBINPATH}/windeployqt ${CMAKE_BINARY_DIR}/bin/CameraSample.exe
    )
endif()

if(WIN32)

    set(OTHER_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../OtherLibs)
    set(FASTVIDEOPATH ${OTHER_LIB_PATH}/FastvideoSDK)
    set(FASTVIDEO_SDK ${FASTVIDEOPATH}/fastvideo_sdk)

    set(COPYFILES
        ${OTHER_LIB_PATH}/ffmpeg/bin/avcodec-57.dll
        ${OTHER_LIB_PATH}/ffmpeg/bin/avdevice-57.dll
        ${OTHER_LIB_PATH}/ffmpeg/bin/avformat-57.dll
        ${OTHER_LIB_PATH}/ffmpeg/bin/avutil-55.dll
        ${OTHER_LIB_PATH}/ffmpeg/bin/swresample-2.dll
        ${OTHER_LIB_PATH}/ffmpeg/bin/swscale-4.dll
        ${FASTVIDEO_SDK}/bin/x64/fastvideo_sdk.dll
    )

    if(USE_XIMEA)
        set(COPYFILES ${COPYFILES}
            ${Ximea_PATH}/x64/xiapi64.dll
        )
    endif()

    function(copy_file PATH FILE)
        add_custom_command( TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${PROJECT_NAME}>/${PATH}
            COMMAND ${CMAKE_COMMAND} -E copy
                ${FILE}
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            )
    endfunction()

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND_EXPAND_LISTS)

    foreach(IT ${COPYFILES})
        message(NOTICE "--> ${IT}")
        copy_file("./" ${IT})
    endforeach()

    add_custom_command( TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${PROJECT_NAME}>/platforms
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${PROJECT_NAME}>/styles/
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${PROJECT_NAME}>/iconengines/
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${PROJECT_NAME}>/imageformats/
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:Qt${QT_VERSION_MAJOR}::Svg>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:Qt${QT_VERSION_MAJOR}::QWindowsIntegrationPlugin>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/platforms/
        # COMMAND ${CMAKE_COMMAND} -E copy_if_different
        #     $<TARGET_FILE:Qt${QT_VERSION_MAJOR}::QWindowsVistaStylePlugin>
        #     $<TARGET_FILE_DIR:${PROJECT_NAME}>/styles/
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:Qt${QT_VERSION_MAJOR}::QSvgPlugin>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/imageformats/
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:Qt${QT_VERSION_MAJOR}::QJpegPlugin>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/imageformats/
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:Qt${QT_VERSION_MAJOR}::QTiffPlugin>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/imageformats/
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:Qt${QT_VERSION_MAJOR}::QTgaPlugin>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/imageformats/
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:Qt${QT_VERSION_MAJOR}::QSvgIconPlugin>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/iconengines/
        COMMAND_EXPAND_LISTS)
endif()
