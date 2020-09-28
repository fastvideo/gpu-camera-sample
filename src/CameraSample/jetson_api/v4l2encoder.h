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

    enum {
        eH264,
        eHEVC
    };

    void setEncoder(int enc);
    void setInsertSpsPpsAtIdrEnabled(bool val);
    void setIDRInterval(int val);
    void setInsertVuiEnabled(bool enabled);
    void setEnableAllIFrameEncode(bool val);
    void setIFrameInterval(int val);
    void setNumBFrames(int val);
    void setFrameRate(int fps);
    void setBitrate(int bitrate);
    void setNumCaptureBuffers(int val);
    void setNumOutputBuffers(int val);
    bool encodeFrame(uint8_t *buf, int width, int height, userbuffer &output, bool nv12 = false);
    bool encodeFrame3(uint8_t **buf, int width, int height, userbuffer &output);

    // 1 get buffer for copy
    bool getInputBuffers3(uint8_t **data, int *lines, uint32_t width, uint32_t height);
    // 2 put to v4l2
    bool putInputBuffers3();
    // 3 get encoded data
    bool getEncodedData(userbuffer& output);

private:
    std::unique_ptr<v4l2EncoderPrivate> mD;
};

#endif // V4L2ENCODER_H
