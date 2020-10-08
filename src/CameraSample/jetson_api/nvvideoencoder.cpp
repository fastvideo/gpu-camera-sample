#include "nvvideoencoder.h"

#include <string>
#include <chrono>
#include <thread>

#include <string.h>

#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <stdlib.h>

extern "C"{
    #include <sys/mman.h>
    #include <linux/videodev2.h>
    #include <linux/v4l2-controls.h>
    #include <v4l2_nv_extensions.h>
    #include <libv4l2.h>
}
const char* nvhostenc = "/dev/nvhost-msenc";

#define CLEAR(buf) memset(&buf, 0, sizeof(buf))

using namespace v4l2encoder;

NvVideoEncoder::NvVideoEncoder(ET et)
    : mEt(et)
{
    int res = v4l2_open(nvhostenc, O_RDWR);
    mInit = res >= 0;
    if(mInit){
        mFD = res;
    }else{
        return;
    }

    v4l2_capability caps;

    res = v4l2_ioctl(mFD, VIDIOC_QUERYCAP, &caps);

    if(res != 0){
        release();
        return;
    }
    capture_plane.fd = mFD;
    capture_plane.buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

    output_plane.fd = mFD;
    output_plane.buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
}

NvVideoEncoder::~NvVideoEncoder()
{
    release();
}

int NvVideoEncoder::setBitrate(uint32_t bitrate)
{
    mBitrate = bitrate;
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEO_BITRATE;
    control.value = bitrate;

    int ret = v4l2_ioctl(mFD, VIDIOC_S_EXT_CTRLS, &ctrls);
    return ret;
}

int NvVideoEncoder::setFrameRate(uint32_t num, uint32_t den)
{
    mFrameRate = num;
    struct v4l2_streamparm params;

    memset(&params, 0, sizeof(params));
    params.parm.output.timeperframe.numerator = num;
    params.parm.output.timeperframe.denominator = den;
    return output_plane.setStreamParms(params);
}

int NvVideoEncoder::setNumBFrames(uint32_t num)
{
    mNumBFrames = num;

    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEO_B_FRAMES;
    control.value = num;

    int ret = v4l2_ioctl(mFD, VIDIOC_S_EXT_CTRLS, &ctrls);

    return ret;}

int NvVideoEncoder::setIFrameInterval(uint32_t val)
{
    mGopSIze = val;

    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEO_GOP_SIZE;
    control.value = val;

    int ret = v4l2_ioctl(mFD, VIDIOC_S_EXT_CTRLS, &ctrls);

    return ret;
}

int NvVideoEncoder::setCapturePlaneFormat(uint32_t pixfmt, uint32_t width, uint32_t height, uint32_t sizeImage)
{
    mPlaneFormat = pixfmt;
    mWidthCapture = width;
    mHeightCapture = height;

    struct v4l2_format format;
    memset(&format, 0, sizeof(format));

    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    format.fmt.pix_mp.pixelformat = pixfmt;
    format.fmt.pix_mp.width = width;
    format.fmt.pix_mp.height = height;
    format.fmt.pix_mp.num_planes = 1;
    format.fmt.pix_mp.plane_fmt[0].sizeimage = sizeImage;

    return capture_plane.setFormat(format);
}

int NvVideoEncoder::setOutputPlaneFormat(uint32_t pixfmt, uint32_t width, uint32_t height)
{
    mPixFmt = pixfmt;
    mWidthOut = width;
    mHeightOut = height;

    int numplanes = 1;
    switch (pixfmt) {
    case V4L2_PIX_FMT_YUV420:
    case V4L2_PIX_FMT_YUV420M:
    case V4L2_PIX_FMT_YUV444:
    case V4L2_PIX_FMT_YUV444M:
    case V4L2_PIX_FMT_YUV422M:
    default:
        numplanes = 3;
        break;
    case V4L2_PIX_FMT_NV12:
    case V4L2_PIX_FMT_NV12M:
    case V4L2_PIX_FMT_NV21:
    case V4L2_PIX_FMT_NV21M:
    case V4L2_PIX_FMT_P010M:
    case V4L2_PIX_FMT_P010:
        numplanes = 2;
        break;
    }

    struct v4l2_format format;
    memset(&format, 0, sizeof(format));
    format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    format.fmt.pix_mp.pixelformat = pixfmt;
    format.fmt.pix_mp.width = width;
    format.fmt.pix_mp.height = height;
    format.fmt.pix_mp.num_planes = numplanes;

    return output_plane.setFormat(format);
}

int NvVideoEncoder::setProfile(uint32_t val)
{
    mProfile = val;
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = mEt == etH264? V4L2_CID_MPEG_VIDEO_H264_PROFILE : V4L2_CID_MPEG_VIDEO_H265_PROFILE;
    control.value = val;

    int ret = v4l2_ioctl(mFD, VIDIOC_S_EXT_CTRLS, &ctrls);

    return ret;
}

int NvVideoEncoder::setLevel(uint32_t val)
{
    mLevel = val;

    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = mEt == etH264? V4L2_CID_MPEG_VIDEO_H264_LEVEL : V4L2_CID_MPEG_VIDEOENC_H265_LEVEL;
    control.value = val;

    int ret = v4l2_ioctl(mFD, VIDIOC_S_EXT_CTRLS, &ctrls);

    return ret;
}

int NvVideoEncoder::setExtControls(v4l2_ext_controls &ctl)
{
    int ret;

    ret = v4l2_ioctl(mFD, VIDIOC_S_EXT_CTRLS, &ctl);

    if (ret < 0)
    {
        std::cout << ("Error setting controls\n");
    }
    else
    {
        //COMP_DEBUG_MSG("Set controls");
    }
    return ret;
}

int NvVideoEncoder::setEnableAllIFrameEncode(bool val){
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    //        RETURN_ERROR_IF_FORMATS_NOT_SET();
    //        RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_ENABLE_ALLIFRAME_ENCODE;
    control.value = val;

    return setExtControls(ctrls);
}

int NvVideoEncoder::setInsertSpsPpsAtIdrEnabled(bool val)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    //RETURN_ERROR_IF_FORMATS_NOT_SET();
    //RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_INSERT_SPS_PPS_AT_IDR;
    control.value = val;

    return setExtControls(ctrls);
}

int NvVideoEncoder::setIDRInterval(int val)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

//    RETURN_ERROR_IF_FORMATS_NOT_SET();
//    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEO_IDR_INTERVAL;
    control.value = val;

    return setExtControls(ctrls);
}

int NvVideoEncoder::setInsertVuiEnabled(bool enabled)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

//     RETURN_ERROR_IF_FORMATS_NOT_SET();
//     RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_INSERT_VUI;
    control.value = enabled;

    return setExtControls(ctrls);
}

int NvVideoEncoder::forceIDR()
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    //RETURN_ERROR_IF_FORMATS_NOT_SET();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_MFC51_VIDEO_FORCE_FRAME_TYPE;

    return (setExtControls(ctrls));
}

void NvVideoEncoder::release()
{
    capture_plane.release();
    output_plane.release();
    if(mFD){
        v4l2_close(mFD);
        mFD = 0;
    }
}

NvVideoEncoder *NvVideoEncoder::createVideoEncoder(const char *name, ET et)
{
    return new NvVideoEncoder(et);
}

///////////////////////////////////////

void NvPlane::release()
{
    buffers.clear();
}

int NvPlane::setupPlane(v4l2_memory typemem, int numbuf, bool f1, bool f2)
{
    int ret;
    memory_type = typemem;
    ret = reqbufs(typemem, numbuf);
    for(int i = 0; i < numbuf; ++i){
        if(typemem == V4L2_MEMORY_MMAP){
            if(queryBuffer(i)){
                return -1;
            }
            if(exportBuffer(i)){
                return -1;
            }
            buffers[i].map();
        }else if(typemem == V4L2_MEMORY_USERPTR){
            buffers[i].allocateMemory();
        }
    }
    return ret;
}

int NvPlane::reqbufs(v4l2_memory mem_type, uint32_t num)
{
    struct v4l2_requestbuffers req;

    CLEAR(req);

    req.count = num;
    req.type = buf_type;
    req.memory = V4L2_MEMORY_MMAP;

    int ret = v4l2_ioctl(fd, VIDIOC_REQBUFS, &req);
    if(ret != 0){
        return ret;
    }

    int id = 0;
    buffers.resize(num);
    for(mapbuffer& buf: buffers){
        buf.buf_type = buf_type;
        buf.mem_type = mem_type;
        buf.n_planes = n_planes;
        buf.id = id++;
        buf.setpixfmt(planefmts);
    }

    return ret;
}

int NvPlane::waitForDQThread(uint32_t ms)
{
    return 0;
}

int NvPlane::setFormat(v4l2_format &format)
{
    pixfmt = format.fmt.pix_mp.pixelformat;
    width = format.fmt.pix_mp.width;
    height = format.fmt.pix_mp.height;

    format.type = buf_type;
    int ret = v4l2_ioctl(fd, VIDIOC_S_FMT, &format);
    if(ret == 0){
        n_planes = format.fmt.pix_mp.num_planes;
        for(int i = 0; i < n_planes; ++i){
            planefmts[i].height = height;
            planefmts[i].bytesperline = format.fmt.pix_mp.plane_fmt[i].bytesperline;
            planefmts[i].sizeimage = format.fmt.pix_mp.plane_fmt[i].sizeimage;
        }
    }
    return ret;
}

int NvPlane::setStreamParms(v4l2_streamparm &parm)
{
    parm.type = buf_type;
    int ret = v4l2_ioctl(fd, VIDIOC_S_PARM, &parm);
    return ret;
}

int NvPlane::setStreamStatus(bool status)
{
    int ret;
    if(status)
        ret = v4l2_ioctl(fd, VIDIOC_STREAMON, &buf_type);
    else
        ret = v4l2_ioctl(fd, VIDIOC_STREAMOFF, &buf_type);
    return ret;
}

int NvPlane::queryBuffer(uint i)
{
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];
    int ret;
    uint32_t j;

    memset(&v4l2_buf, 0, sizeof(struct v4l2_buffer));
    memset(planes, 0, sizeof(planes));
    v4l2_buf.index = i;
    v4l2_buf.type = buf_type;
    v4l2_buf.memory = memory_type;
    v4l2_buf.m.planes = planes;
    v4l2_buf.length = n_planes;

    ret = v4l2_ioctl(fd, VIDIOC_QUERYBUF, &v4l2_buf);
    if (ret){
        return ret;
    }
    else
    {
        for (j = 0; j < v4l2_buf.length; j++)
        {
            buffers[i].planes[j].length = v4l2_buf.m.planes[j].length;
            buffers[i].planes[j].mem_offset =
                v4l2_buf.m.planes[j].m.mem_offset;
        }
    }

    return ret;
}

int NvPlane::exportBuffer(uint i)
{
    struct v4l2_exportbuffer expbuf;
    int ret;
    int j;

    memset(&expbuf, 0, sizeof(expbuf));
    expbuf.type = buf_type;
    expbuf.index = i;

    for(j = 0; j < n_planes; ++j){
        expbuf.plane = j;
        ret = v4l2_ioctl(fd, VIDIOC_EXPBUF, &expbuf);
        if (ret){
            return -1;
        }else{
            buffers[i].planes[j].fd = expbuf.fd;
        }
    }
    return 0;
}

int NvPlane::qBuffer(v4l2_buffer &buf)
{
    buf.type = buf_type;
    buf.memory = memory_type;
    buf.length = n_planes;

    mapbuffer &buffer = buffers[buf.index];

    if(memory_type == V4L2_MEMORY_USERPTR){
        for(int i = 0; i < n_planes; ++i){
            buf.m.planes[i].m.userptr = (unsigned long)buffer.planes[i].buf;
            buf.m.planes[i].bytesused = buffer.planes[i].bytesused;
        }
    }
    if(memory_type == V4L2_MEMORY_MMAP){
        for(int i = 0; i < n_planes; ++i){
            buf.m.planes[i].bytesused = buffer.planes[i].bytesused;
        }
    }
    int ret = v4l2_ioctl(fd, VIDIOC_QBUF, &buf);

    //std::cout << "qbuffer " << ret << std::endl;

    return ret;
}

int NvPlane::dqBuffer(v4l2_buffer &buf, mapbuffer **_buffer)
{

    buf.type = buf_type;
    buf.memory = memory_type;

    int ret = 0;

    do{
        ret = v4l2_ioctl(fd, VIDIOC_DQBUF, &buf);

        //std::cout << "dqbuffer " <<  ret << " " <<  buf.index << std::endl;

        if(ret == 0){
            mapbuffer &buffer = buffers[buf.index];
            if(_buffer)
                *_buffer = &buffer;
            for(int i = 0; i < buffer.n_planes; ++i){
                buffer.planes[i].bytesused = buf.m.planes[i].bytesused;
            }
        }else if(ret == EAGAIN){
            break;
        }else{
            break;
        }
    }while(ret);
    return ret;
}

///////////////////////

mapbuffer::mapbuffer()
{

}

int mapbuffer::setpixfmt(Plane planefmts[])
{
    for(int i = 0; i < n_planes; ++i){
        planes[i].fmt = planefmts[i];
    }
    return 0;
}

int mapbuffer::setpixfmt(uint32_t pixfmt, int width, int height)
{
    switch (pixfmt) {
    case V4L2_PIX_FMT_YUV444M:
        n_planes = 3;
        planes[0].fmt.width = width;
        planes[1].fmt.width = width;
        planes[2].fmt.width = width;

        planes[0].fmt.height = height;
        planes[1].fmt.height = height;
        planes[2].fmt.height = height;

        planes[0].fmt.bytesperpixel = 1;
        planes[1].fmt.bytesperpixel = 1;
        planes[2].fmt.bytesperpixel = 1;
        break;
    case V4L2_PIX_FMT_YUV422M:
        n_planes = 3;
        planes[0].fmt.width = width;
        planes[1].fmt.width = width/2;
        planes[2].fmt.width = width/2;

        planes[0].fmt.height = height;
        planes[1].fmt.height = height;
        planes[2].fmt.height = height;

        planes[0].fmt.bytesperpixel = 1;
        planes[1].fmt.bytesperpixel = 1;
        planes[2].fmt.bytesperpixel = 1;
        break;
    case V4L2_PIX_FMT_YUV420M:
        n_planes = 3;
        planes[0].fmt.width = width;
        planes[1].fmt.width = width/2;
        planes[2].fmt.width = width/2;

        planes[0].fmt.height = height;
        planes[1].fmt.height = height/2;
        planes[2].fmt.height = height/2;

        planes[0].fmt.bytesperpixel = 1;
        planes[1].fmt.bytesperpixel = 1;
        planes[2].fmt.bytesperpixel = 1;
        break;
    case V4L2_PIX_FMT_NV12M:
        n_planes = 2;
        planes[0].fmt.width = width;
        planes[1].fmt.width = width/2;

        planes[0].fmt.height = height;
        planes[1].fmt.height = height/2;

        planes[0].fmt.bytesperpixel = 1;
        planes[1].fmt.bytesperpixel = 2;
        break;
    case V4L2_PIX_FMT_GREY:
        n_planes = 1;
        planes[0].fmt.width = width;

        planes[0].fmt.height = height;

        planes[0].fmt.bytesperpixel = 1;
        break;
    default:
        break;
    }
    return 0;
}

int mapbuffer::map()
{
    int ret = 0;
    release();

    for(size_t i = 0; i < n_planes; ++i){
        planes[i].buf = (uint8_t*)mmap(nullptr, planes[i].length,
                                       PROT_READ | PROT_WRITE, MAP_SHARED,
                                       planes[i].fd, planes[i].mem_offset);
    }

    return ret;
}

int mapbuffer::allocateMemory()
{
    for(int i = 0; i < n_planes; ++i){
        Buffer &p = planes[i];
        Plane &f = p.fmt;
        uint32_t sz = f.sizeimage;
        sz = std::max(sz, f.width * f.height * f.bytesperpixel);
        p.ubuf.resize(sz);
        p.buf = p.ubuf.data();
    }
    return 0;
}

void mapbuffer::release()
{
    if(mem_type == V4L2_MEMORY_USERPTR){
        for(int i = 0; i < n_planes; ++i){
            planes[i].buf = nullptr;
            planes[i].ubuf.clear();
        }
    }
    if(mem_type == V4L2_MEMORY_MMAP){
        for(int i = 0; i < n_planes; ++i){
            if(planes[i].buf)
                munmap(planes[i].buf, planes[i].length);
            planes[i].buf = nullptr;
        }
    }
}
