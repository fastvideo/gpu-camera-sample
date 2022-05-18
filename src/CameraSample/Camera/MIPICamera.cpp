#include "MIPICamera.h"

#include <cuda.h>

#include "RawProcessor.h"

#include <iostream>
#include <libv4l2.h>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/mman.h>

#include <thread>

union BS{
    uint32_t ui;
    char c[4];
};

/**
  $ v4l2-ctl -d /dev/video0 --list-ctrls
  fuse_id 0x009a2007 (str)    : min=0 max=16 step=2 value='9332a83b82c0919c' flags=read-only, has-payload
                           gain 0x009a2009 (int64)  : min=10 max=160 step=1 default=10 value=10 flags=slider
                       exposure 0x009a200a (int64)  : min=34 max=550385 step=1 default=33334 value=34 flags=slider
                     frame_rate 0x009a200b (int64)  : min=1816577 max=30000000 step=1 default=30000000 value=1816577 flags=slider
                 exposure_short 0x009a200c (int64)  : min=34 max=550385 step=1 default=33334 value=34 flags=slider
                    bypass_mode 0x009a2064 (intmenu): min=0 max=1 default=0 value=0
                override_enable 0x009a2065 (intmenu): min=0 max=1 default=0 value=0
                   height_align 0x009a2066 (int)    : min=1 max=16 step=1 default=1 value=1
                     size_align 0x009a2067 (intmenu): min=0 max=2 default=0 value=0
               write_isp_format 0x009a2068 (int)    : min=1 max=1 step=1 default=1 value=1
       sensor_signal_properties 0x009a2069 (u32)    : min=0 max=4294967295 step=1 default=0 [30][18] flags=read-only, has-payload
        sensor_image_properties 0x009a206a (u32)    : min=0 max=4294967295 step=1 default=0 [30][16] flags=read-only, has-payload
      sensor_control_properties 0x009a206b (u32)    : min=0 max=4294967295 step=1 default=0 [30][36] flags=read-only, has-payload
              sensor_dv_timings 0x009a206c (u32)    : min=0 max=4294967295 step=1 default=0 [30][16] flags=read-only, has-payload
               low_latency_mode 0x009a206d (bool)   : default=0 value=0
               preferred_stride 0x009a206e (int)    : min=0 max=65535 step=1 default=0 value=0
                   sensor_modes 0x009a2082 (int)    : min=0 max=30 step=1 default=30 value=3 flags=read-only

*/

struct Mem{
    int rows = 0;
    int cols = 0;
    int depth = 0;
    std::vector<uint8_t> data;
    int step[2] = {0, 0};

    Mem(){}
    Mem(int rows, int cols, int depth){
        resize(rows, cols, depth);
    }

    size_t size() const{
        return data.size();
    }

    uint8_t* ptr(int y){
        return data.data() + y * cols * depth;
    }
    const uint8_t* ptr(int y) const{
        return data.data() + y * cols * depth;
    }

    void resize(int rows, int cols, int depth){
        this->rows = rows;
        this->cols = cols;
        this->depth = depth;
        step[1] = depth;
        step[0] = cols * depth;
        this->data.resize(this->rows * this->cols * this->depth);

    }
    static Mem zeros(int rows, int cols, int depth){
        Mem res;
        res.resize(rows, cols, depth);
        memset(res.data.data(), 0, res.data.size());
    }
};

struct Res{
    int w = 0;
    int h = 0;
    Res(){

    }
    Res(int w, int h){
        this->w = w;
        this->h = h;
    }
};

const Res Resolutions[] = {
    Res(2592, 1944),
    Res(2592, 1458),
    Res(640, 480),
};

inline uint32_t make_fmt(char* fmt)
{
    uint32_t res = fmt[0] | (fmt[1] << 8) | (fmt[2] << 16) | (fmt[3] << 24);
    //memcpy(&res, fmt, 4);
    return res;
}

#define CLEAR(fmt) memset(&(fmt), 0, sizeof(fmt))

struct buffer{
    void *start = nullptr;
    size_t length = 0;
};

class CameraV4l2
{
public:
    enum {
        EXPOSURE = 0x009a200a,
    };

    CameraV4l2(const std::string &dev = "/dev/video0"){
        mDev = dev;
    }
    ~CameraV4l2(){
        close();
    }

    void setDev(const std::string& dev){
        mDev = dev;
    }
    void setDev(int dev){
        mDev = "/dev/video" + std::to_string(dev);
    }

    int width() const{
        return mWidth;
    }
    int height() const{
        return mHeight;
    }
    float fps() const{
        return mFps;
    }

    bool is_open() const{
        return mIsOpen;
    }
    bool open(){
        std::lock_guard<std::mutex> guard(mMutex);

        if(mIsOpen)
            return true;

        mFd = ::open(mDev.c_str(), O_RDWR | O_NONBLOCK);

        if(mFd < 0)
            return false;

        Res r = Resolutions[mResolutionId];

        struct v4l2_format      fmt;
        CLEAR(fmt);
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(mFd, VIDIOC_G_FMT, &fmt);
        uint32_t pixfmt = fmt.fmt.pix.pixelformat;
        
        CLEAR(fmt);
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = r.w;
        fmt.fmt.pix.height = r.h;
        fmt.fmt.pix.pixelformat = pixfmt;//make_fmt("BG10");//0x30314742;//01GB;
        fmt.fmt.pix.field = V4L2_FIELD_ANY;
        xioctl(mFd, VIDIOC_S_FMT, &fmt);
        mWidth = fmt.fmt.pix.width;
        mHeight = fmt.fmt.pix.height;
        mBytesPerLines =fmt.fmt.pix.bytesperline;

        struct v4l2_frmivalenum temp;
        CLEAR(temp);
        temp.pixel_format = fmt.fmt.pix.pixelformat;
        temp.width = mWidth;
        temp.height = mHeight;
        xioctl(mFd, VIDIOC_ENUM_FRAMEINTERVALS, &temp);
        if (temp.type == V4L2_FRMIVAL_TYPE_DISCRETE) {
             while (xioctl(mFd, VIDIOC_ENUM_FRAMEINTERVALS, &temp)) {
                std::cout << float(temp.discrete.denominator)/temp.discrete.numerator << " fps" << endl;
                mFps = float(temp.discrete.denominator)/temp.discrete.numerator;
                temp.index += 1;
            }
        }

        BS bs;
        bs.ui = fmt.fmt.pix.pixelformat;

        struct v4l2_requestbuffers req;
        CLEAR(req);
        req.count = 2;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        xioctl(mFd, VIDIOC_REQBUFS, &req);

        mBuffrs.resize(req.count);
        for(size_t i = 0; i < mBuffrs.size(); ++i){
            struct v4l2_buffer buf;
            CLEAR(buf);
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            xioctl(mFd, VIDIOC_QUERYBUF, &buf);

            mBuffrs[i].length = buf.length;
            mBuffrs[i].start = v4l2_mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, mFd, buf.m.offset);
            if (MAP_FAILED == mBuffrs[i].start){
                return false;
            }
        }

        for(size_t i = 0; i < mBuffrs.size(); ++i){
            struct v4l2_buffer buf;
            CLEAR(buf);
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            xioctl(mFd, VIDIOC_QBUF, &buf);
        }

        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(mFd, VIDIOC_STREAMON, &type);

        mIsOpen = mFd >= 0;

        return mIsOpen;
    }
    void close(){
        mIsOpen = false;

        std::lock_guard<std::mutex> guard(mMutex);

        if(mFd >= 0){

            int res = 0;
            for(size_t i = 0; i < mBuffrs.size(); ++i){
                 res = v4l2_munmap(mBuffrs[i].start, mBuffrs[i].length);
            }
            mBuffrs.clear();

            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            xioctl(mFd, VIDIOC_STREAMOFF, &type);
            ::close(mFd);
            mFd = 0;
        }
    }
    void get(Mem &res){
        res.resize(mHeight, mWidth, 2);

        if(!is_open())
            return;

        if(mBuffrs.empty())
            return;

        std::lock_guard<std::mutex> guard(mMutex);

        v4l2_buffer buf;
        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        xioctl(mFd, VIDIOC_DQBUF, &buf);

        if(mBuffrs.empty() || buf.index > mBuffrs.size())
            return;

        for(int i = 0; i < mHeight; ++i){
            char *ptr = (char*)mBuffrs[buf.index].start + i * mBytesPerLines;
            memcpy(res.ptr(i), ptr, res.step[0]);
        }
        //memcpy(res.data, mBuffrs[buf.index].start, buf.bytesused);

        xioctl(mFd, VIDIOC_QBUF, &buf);
    }

    float exposure() const {
        struct v4l2_ext_control ctrl;
        CLEAR(ctrl);
        ctrl.id = EXPOSURE;
        ctrl.value64 = 0;

        v4l2_ext_controls ctrls;
        CLEAR(ctrls);
        ctrls.ctrl_class = V4L2_CTRL_ID2CLASS(ctrl.id);
        ctrls.count = 1;
        ctrls.controls = &ctrl;

        if(xioctl(mFd, VIDIOC_G_EXT_CTRLS, &ctrls)){
            std::cout << ctrls.count << std::endl;
            return ctrl.value64;
        }
        return mExposure;
    }

    void set_exposure(int val){
        if(!mIsOpen)
            return;
        mExposure = val;

        std::lock_guard<std::mutex> guard(mMutex);

        struct v4l2_ext_control ctrl;
        CLEAR(ctrl);
        ctrl.id = EXPOSURE;
        ctrl.value64 = val;

        v4l2_ext_controls ctrls;
        CLEAR(ctrls);
        ctrls.ctrl_class = V4L2_CTRL_ID2CLASS(ctrl.id);
        ctrls.count = 1;
        ctrls.controls = &ctrl;

        xioctl(mFd, VIDIOC_S_EXT_CTRLS, &ctrls);
    }

    void set_resolution_id(int val){
        if(val == mResolutionId){
            return;
        }
        mResolutionId = val;
        close();
        //std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        open();
    }

private:
    std::string mDev;
    int mFd = 0;
    bool mIsOpen = false;
    int mWidth = 0;
    int mHeight = 0;
    float mFps = 0;
    std::vector<buffer> mBuffrs;
    int mBytesPerLines = 0;
    std::mutex mMutex;
    int mExposure = 0;

    int mResolutionId = 0;

    bool xioctl(int fd, int request, void *args) const{
        int r;

        do{
            r = ioctl(fd, request, args);
        }while(r == -1 && ((errno == EINTR) || (errno == EAGAIN)));

        if(r == -1){
            std::cout << r << " " << errno << std::endl;
            return false;
        }
        return true;
    }
};

///////////////////////////////////////

MIPICamera::MIPICamera()
{
    mCamera.reset(new CameraV4l2);

    mCameraThread.setObjectName(QStringLiteral("MIPICameraThread"));
    moveToThread(&mCameraThread);
    mCameraThread.start();
}

MIPICamera::~MIPICamera()
{
    mCameraThread.quit();
    mCameraThread.wait(3000);

    mCamera.reset();
}

void MIPICamera::startStreaming()
{
    if(mState != cstStreaming)
        return;

    QByteArray frameData;
    frameData.resize((int)mInputBuffer.size());

    Mem image;

    QElapsedTimer tmr;
    while(mState == cstStreaming)
    {
        tmr.restart();
        mCamera->get(image);
        unsigned char* dst = mInputBuffer.getBuffer();
        cudaMemcpy(dst, image.ptr(0), image.size(), cudaMemcpyHostToDevice);
        mInputBuffer.release();

        {
            QMutexLocker l(&mLock);
            mRawProc->acqTimeNsec = tmr.nsecsElapsed();
            mRawProc->wake();
        }
    }
    mCamera->close();
}

bool MIPICamera::open(uint32_t devID)
{
    mCamera->setDev(devID);
    bool res = mCamera->open();
    if(res){
        mWidth = mCamera->width();
        mHeight = mCamera->height();
        mSurfaceFormat = FAST_I12;
        mImageFormat = cif12bpp;
        mWhite = 4095;
        mBblack = 0;
        mIsColor = true;
        mFPS = mCamera->fps();
        mPattern = FAST_BAYER_BGGR;
        if(!mInputBuffer.allocate(mWidth, mHeight, mSurfaceFormat))
            return false;
    }

    mDevID = devID;
    mState = cstStopped;
    emit stateChanged(cstStopped);
    return res;
}

bool MIPICamera::start()
{
    mState = cstStreaming;
    emit stateChanged(cstStreaming);
    QTimer::singleShot(0, this, [this](){startStreaming();});
    return true;
}

bool MIPICamera::stop()
{
    mState = cstStopped;
    emit stateChanged(cstStopped);

    return true;
}

void MIPICamera::close()
{
    stop();
    mCamera->close();
    //xiCloseDevice(hDevice);
    mState = cstClosed;
    emit stateChanged(cstClosed);
}

bool MIPICamera::getParameter(GPUCameraBase::cmrCameraParameter param, float &val)
{
    if(param < 0 || param > prmLast)
        return false;

    switch (param)
    {
    case prmFrameRate:
        val = mCamera->fps();
        return true;

    case prmExposureTime:
        val = mCamera->exposure();
        return true;

    default:
        break;
    }

    return false;
}

bool MIPICamera::setParameter(GPUCameraBase::cmrCameraParameter param, float val)
{
    if(param < 0 || param > prmLast)
        return false;

    int v = 0;
    switch (param)
    {
    case prmFrameRate:
        if(val < 0)
        {
        }
        else
        {
        }
        break;

    case prmExposureTime:
        mCamera->set_exposure(val);
        return true;

    default:
        break;
    }

    return false;
}

bool MIPICamera::getParameterInfo(GPUCameraBase::cmrParameterInfo &info)
{
    if(info.param < 0 || info.param > prmLast)
        return false;
    switch(info.param)
    {
    case prmFrameRate:
        info.min = mCamera->fps();
        info.max = mCamera->fps();
        info.increment = 0;
        break;

    case prmExposureTime:
        info.min = 0;
        info.max = 10000000000;
        info.increment = 1;
        break;

    default:
        break;
    }

    return true;
}
