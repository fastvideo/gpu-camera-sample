INCLUDEPATH += $$PWD

HEADERS += \
    $$PWD/NvBufSurface.h \
    $$PWD/v4l2_nv_extensions.h \
    $$PWD/nvvideoencoder.h \
    $$PWD/v4l2encoder.h \
    $$PWD/common_types.h \
    #$$PWD/nvbuf_utils.h

SOURCES += \
    $$PWD/nvvideoencoder.cpp \
    $$PWD/v4l2encoder.cpp

TARGET_ARCH=$${QT_ARCH}


contains(TARGET_ARCH, "arm64"){
    TEGRA_ARMABI = aarch64-linux-gnu
}
LIBS += -L/usr/lib/$$TEGRA_ARMABI/  \
        -L/usr/lib/$$TEGRA_ARMABI/tegra/ \
        -lnvbufsurface \
        -lv4l2 #-lnvbuf_utils


