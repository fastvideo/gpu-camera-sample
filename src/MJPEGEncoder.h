#ifndef MJPEGENCODER_H
#define MJPEGENCODER_H

#include <QString>
#include <QMutex>
#include "fastvideo_sdk.h"

struct AVFormatContext;
struct AVFrame;

class MJPEGEncoder
{
public:
    MJPEGEncoder(int width,
                 int height,
                 int fps,
                 fastJpegFormat_t fmt,
                 const QString& outFileName);
    ~MJPEGEncoder();
    bool isOpened(){return mErr >= 0;}
    bool addJPEGFrame(unsigned char *jpgPtr, int jpgSize);
    void close();
private:
    AVFormatContext* mFmtCtx = nullptr;
    int mFramesProcessed = 0;
    int mErr = 0;

    QMutex mLock;
};

#endif // MJPEGENCODER_H
