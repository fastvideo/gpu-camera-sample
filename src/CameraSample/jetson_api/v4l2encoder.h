#ifndef V4L2ENCODER_H
#define V4L2ENCODER_H

#include <vector>
#include <memory>

#include "common_types.h"

class v4l2EncoderPrivate;

class v4l2Encoder
{
public:
    v4l2Encoder();
    ~v4l2Encoder();

    void setIFrameInterval(int val);
    void setNumBFrames(int val);
    void setFrameRate(int fps);
    void setBitrate(int bitrate);
    void setNumCaptureBuffers(int val);
    void setNumOutputBuffers(int val);
    bool encodeFrame(uint8_t *buf, int width, int height, userbuffer &output, bool nv12 = false);

private:
    std::unique_ptr<v4l2EncoderPrivate> mD;
};

#endif // V4L2ENCODER_H
