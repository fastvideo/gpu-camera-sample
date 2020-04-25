include(../../common_defs.pri)
include(../../common_funcs.pri)
win32: include(../../common.pri)
unix:  include(../../common_unix.pri)

INCLUDEPATH += $$PWD

HEADERS += \
    $$PWD/cuviddecoder.h

SOURCES += \
    $$PWD/cuviddecoder.cpp

