DEFINES += SUPPORT_XIMEA
#DEFINES += SUPPORT_FLIR
#DEFINES += SUPPORT_IMPERX
#DEFINES += SUPPORT_GENICAM

TARGET_ARCH=$${QT_ARCH}
BITS = 64
contains(TARGET_ARCH, arm64) {
    PLATFORM = Arm64
    GENAPIVER = gcc49_v3_2
} else {
    win32: {
        PLATFORM = x64
        GENAPIVER = VC141_v3_2

        #For Imperx cameras support
        #GENAPIVER = VC140_v3_0
    }
    unix: {
        PLATFORM = Linux$$BITS
        GENAPIVER = gcc48_v3_2

        # For Imperx cameras support
        #GENAPIVER = gcc421_v3_0
    }
}

PROJECT_NAME = GPUCameraSample

DESTDIR = $$absolute_path($$PWD/../$${PROJECT_NAME}_$$PLATFORM)
CONFIG(debug, debug|release){
    DESTDIR = $$DESTDIR/debug
}else:CONFIG(release, debug|release){
    DESTDIR = $$DESTDIR/release
}
