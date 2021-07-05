#DEFINES += SUPPORT_XIMEA
#DEFINES += SUPPORT_FLIR
#DEFINES += SUPPORT_IMPERX
#DEFINES += SUPPORT_GENICAM

#GENAPIVER = VC140_v3_0
GENAPIVER = VC141_v3_2

TARGET_ARCH=$${QT_ARCH}
BITS = 64
contains(TARGET_ARCH, arm64 ) {
    PLATFORM = Arm64
} else {
    win32: PLATFORM = x64
    unix: PLATFORM = Linux$$BITS
}

PROJECT_NAME = GPUCameraSample

DESTDIR = $$absolute_path($$PWD/../$${PROJECT_NAME}_$$PLATFORM)
CONFIG(debug, debug|release){
    DESTDIR = $$DESTDIR/debug
}else:CONFIG(release, debug|release){
    DESTDIR = $$DESTDIR/release
}
