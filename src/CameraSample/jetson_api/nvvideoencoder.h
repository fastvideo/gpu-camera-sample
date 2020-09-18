#ifndef NVVIDEOENCODER_H
#define NVVIDEOENCODER_H

#include <iostream>
#include <vector>

#include "common_types.h"

extern "C"{
    #include <linux/videodev2.h>
    #include <linux/v4l2-controls.h>
    #include <v4l2_nv_extensions.h>
    #include <libv4l2.h>
}

namespace v4l2encoder{

struct Plane{
    uint32_t bytesperline = 0;
    uint32_t sizeimage = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t bytesperpixel = 1;
};

struct Buffer{
    int fd = -1;
    uint8_t *buf = nullptr;
    uint32_t length = 0;
    uint32_t mem_offset = 0;
    size_t bytesused = 0;
    userbuffer ubuf;
    Plane fmt;
};

struct mapbuffer{
    int id = 0;
    uint32_t buf_type = 0;
    uint32_t mem_type = 0;
    uint32_t n_planes = 0;
    Buffer planes[MAX_PLANES];

    mapbuffer();
    ~mapbuffer() { release(); }

    int setpixfmt(Plane planefmts[]);
    int setpixfmt(uint32_t pixfmt, int width, int height);
    int map();
    int allocateMemory();
    void release();
};

struct NvPlane{
    int fd;
    int buf_type = 0;
    int memory_type = 0;
    int n_planes = 0;
    int pixfmt = 0;
    int width = 0;
    int height = 0;

    Plane planefmts[MAX_PLANES];

    void release();

    int setupPlane(v4l2_memory typemem, int numbuf, bool f1, bool f2);
    int reqbufs(enum v4l2_memory mem_type, uint32_t num);
    int waitForDQThread(uint32_t ms);
    int setFormat(struct v4l2_format& format);
    int setStreamParms(struct v4l2_streamparm &parm);
    int setStreamStatus(bool status);

    int queryBuffer(uint i);
    int exportBuffer(uint i);

    uint32_t getNumBuffers() const { return buffers.size(); }
    mapbuffer* getNthBuffer(uint32_t num){ return &buffers[num]; }

    int qBuffer(struct v4l2_buffer& buf);
    int dqBuffer(struct v4l2_buffer& buf, mapbuffer **_buffer);

    std::vector<mapbuffer> buffers;
};

class NvVideoEncoder
{
public:
    enum ET{
        etH264,
        etH265
    };

    NvVideoEncoder(ET et = etH264);
    ~NvVideoEncoder();

    int setBitrate(uint32_t bitrate);
    int setFrameRate(uint32_t num, uint32_t den);
    int setNumBFrames(uint32_t num);
    int setIFrameInterval(uint32_t val);
    int setCapturePlaneFormat(uint32_t pixfmt, uint32_t width, uint32_t height, uint32_t sizeImage);
    int setOutputPlaneFormat(uint32_t pixfmt, uint32_t width, uint32_t height);
    int setProfile(uint32_t val);
    int setLevel(uint32_t val);
    int setExtControls(struct v4l2_ext_controls &ctl);
    int setEnableAllIFrameEncode(bool val);
    int setInsertSpsPpsAtIdrEnabled(bool val);
    int setIDRInterval(int val);
    int setInsertVuiEnabled(bool enabled);
    int forceIDR();

    uint32_t pixFmt() const { return mPixFmt; }

    bool isInit() const { return mInit; }
    void release();

    NvPlane capture_plane;
    NvPlane output_plane;

    static NvVideoEncoder *createVideoEncoder(const char* name, ET et = etH264);

private:
    uint32_t mWidthCapture = 0;
    uint32_t mHeightCapture = 0;
    uint32_t mWidthOut = 0;
    uint32_t mHeightOut = 0;
    uint32_t mBitrate = 10e6;
    uint32_t mPixFmt = 0;
    uint32_t mProfile = 0;
    uint32_t mLevel = 0;
    uint32_t mFrameRate = 60;
    uint32_t mFD = 0;
    uint32_t mPlaneFormat = 0;
    uint32_t mNumBFrames = 0;
    uint32_t mGopSIze = 1;
    bool mInit = false;

    ET mEt = etH264;
};

}

#endif // NVVIDEOENCODER_H
