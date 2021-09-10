#include "cuviddecoder.h"

#include "cuviddec.h"
#include <nvcuvid.h>

#include <memory>
#include <iostream>
#include <list>

//////////////////////////////
int sequenceCallback(void *obj, CUVIDEOFORMAT *params);
int decodeCallback(void *obj, CUVIDPICPARAMS *params);
int displayCallback(void *obj, CUVIDPARSERDISPINFO *params);

class CuvidPrivate{
public:
    CUvideodecoder mDecoder = nullptr;
    CUvideoparser mParser = nullptr;
    CUcontext mContext = nullptr;
    CUvideoctxlock mLock = nullptr;
    unsigned mWidth = 0;
    unsigned mHeight = 0;
    bool mUpdateImage = false;

    CUVIDEOFORMAT mFmt;

    PImage mFrame;
//    std::list<PImage> mFrames;
    size_t mMaximumQueueSize = 2;

    CuvidPrivate(CuvidDecoder::ET et, int device = 0){
        CUresult res;
        CUdevice dev;

        res = cuInit(0);
        if(res != CUDA_SUCCESS)
            return;

        res = cuDeviceGet(&dev, device);
        if(res != CUDA_SUCCESS)
            return;

        res = cuCtxCreate(&mContext, 0, dev);
        if(res != CUDA_SUCCESS)
            return;

        res = cuvidCtxLockCreate(&mLock, mContext);
        if(res != CUDA_SUCCESS)
            return;

        CUVIDPARSERPARAMS params;
        memset(&params, 0, sizeof(params));

        if(et == CuvidDecoder::eH264)
            params.CodecType = cudaVideoCodec_H264;
        else
            params.CodecType = cudaVideoCodec_HEVC;

        params.ulErrorThreshold = 100;
        params.ulClockRate = 0;
        params.ulMaxDisplayDelay = 0;
        params.ulMaxNumDecodeSurfaces = 2;
        params.pUserData = this;
        params.pfnDecodePicture = decodeCallback;
        params.pfnDisplayPicture = displayCallback;
        params.pfnSequenceCallback = sequenceCallback;

        res = cuvidCreateVideoParser(&mParser, &params);
        if(res != CUDA_SUCCESS){
            printf("create parser error %d\n", res);
        }
    }
    ~CuvidPrivate(){
        if(mParser)
            cuvidDestroyVideoParser(mParser);

        release_decoder();

        if(mLock){
            cuvidCtxLockDestroy(mLock);
            mLock = nullptr;
        }

        if(mContext){
            cuCtxDestroy(mContext);
            mContext = nullptr;
        }
    }

    bool decode(uint8_t *data, size_t size){
        if(!mParser)
            return false;

        CUresult res;

        CUVIDSOURCEDATAPACKET packet;
        memset(&packet, 0, sizeof(packet));
        packet.payload = data;
        packet.payload_size = static_cast<unsigned long>(size);

        res = cuvidParseVideoData(mParser, &packet);

        if(res != CUDA_SUCCESS)
            return false;

        return true;
    }

    void init_decoder(){
        CUresult res;

        CUVIDDECODECREATEINFO info;
        memset(&info, 0, sizeof(info));
        info.ulWidth = mWidth;
        info.ulHeight = mHeight;
        info.ulMaxWidth = mWidth;
        info.ulMaxHeight = mHeight;
        info.CodecType = mFmt.codec;
        info.ChromaFormat = mFmt.chroma_format;
        info.OutputFormat = cudaVideoSurfaceFormat_NV12;
        info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
        info.ulTargetWidth = mWidth;
        info.ulTargetHeight = mHeight;
        info.ulNumOutputSurfaces = 1;
        info.ulNumDecodeSurfaces = 1;
        info.ulCreationFlags = cudaVideoCreate_PreferCUDA | cudaVideoCreate_PreferCUVID;
        info.bitDepthMinus8 = mFmt.bit_depth_luma_minus8;

        cuvidCtxLock(mLock, 0);
        res = cuvidCreateDecoder(&mDecoder, &info);
        cuvidCtxUnlock(mLock, 0);
        if(res != CUDA_SUCCESS){
            printf("initialize error %d\n", res);
        }
    }
    void release_decoder(){
        if(mDecoder)
            cuvidDestroyDecoder(mDecoder);
        mDecoder = nullptr;
    }

    void sequence(CUVIDEOFORMAT *fmt){

        int w = fmt->display_area.right - fmt->display_area.left;
        int h = fmt->display_area.bottom - fmt->display_area.top;
        if(!mDecoder || w != mWidth || h != mHeight){
            release_decoder();
        }

        mWidth  = w;
        mHeight = h;
        mFmt = *fmt;

        if(!mDecoder){
            init_decoder();
        }

    }
    void decode(CUVIDPICPARAMS *params){
        if(!mDecoder)
            return;

        CUresult res = cuvidDecodePicture(mDecoder, params);
        if(res != CUDA_SUCCESS){
            printf("decode error %d\n", res);
        }
    }
    void display(CUVIDPARSERDISPINFO *params){
        CUresult res;
        CUVIDPROCPARAMS procParams;
        CUVIDGETDECODESTATUS status;
        memset(&procParams, 0, sizeof(procParams));

        CUdeviceptr srcFrame = 0;
        unsigned srcPitch = 0;
        res = cuvidMapVideoFrame(mDecoder , params->picture_index, &srcFrame, &srcPitch, &procParams);

        res = cuvidGetDecodeStatus(mDecoder, params->picture_index, &status);

//        if(!mFrame.get() || mFrame->width != mWidth || mFrame->height != mHeight)
//            mFrame.reset(new Image);
        if(mFrame.get()){
            mFrame->setNV12(mWidth, mHeight);

            int offset = 0;
            uint8_t *data[2] = {mFrame->yuv.data(), mFrame->yuv.data() + mWidth * mHeight};
            for(int i = 0; i < 2; ++i){
                CUDA_MEMCPY2D cpy;
                memset(&cpy, 0, sizeof(cpy));
                cpy.srcY = offset;

                cpy.WidthInBytes = mWidth;
                cpy.Height = mHeight >> i;

                cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                cpy.srcDevice = srcFrame;
                cpy.srcPitch = srcPitch;

                cpy.dstMemoryType = CU_MEMORYTYPE_HOST;
                cpy.dstPitch = mWidth;
                cpy.dstHost = data[i];

                offset += mHeight;
                res = cuMemcpy2D(&cpy);
            }

     //        size_t sz = srcPitch * mHeight + srcPitch * mHeight/2;
    //        res = cuMemcpyDtoH(image->data.data(), srcFrame, sz);

            if(res == CUDA_SUCCESS){
                printf("image got\n");
                mUpdateImage = true;
            }
        }

        res = cuvidUnmapVideoFrame(mDecoder, srcFrame);
    }
};

//////////////////////////////

int sequenceCallback(void *obj, CUVIDEOFORMAT *params)
{
    CuvidPrivate *that = (CuvidPrivate*)obj;
    that->sequence(params);
    return 1;
}

int decodeCallback(void *obj, CUVIDPICPARAMS *params)
{
    CuvidPrivate *that = (CuvidPrivate*)obj;
    that->decode(params);
    return 1;
}

int displayCallback(void *obj, CUVIDPARSERDISPINFO *params)
{
    CuvidPrivate *that = (CuvidPrivate*)obj;
    that->display(params);
    return 1;
}

//////////////////////////////

CuvidDecoder::CuvidDecoder(ET et)
{
    mD.reset(new CuvidPrivate(et));
}

CuvidDecoder::~CuvidDecoder()
{
    mD.reset();
}

size_t CuvidDecoder::maximumQueueSize()
{
    return mD->mMaximumQueueSize;
}

void CuvidDecoder::setMaximumQueueSize(size_t size)
{
    mD->mMaximumQueueSize = size;
}

bool CuvidDecoder::isUpdateImage() const
{
    return mD->mUpdateImage;
}

void CuvidDecoder::resetUpdateImage()
{
    mD->mUpdateImage = false;
}

bool CuvidDecoder::decode(uint8_t *data, size_t size, PImage& image)
{
    if(!image){
        image.reset(new RTSPImage());
    }
    if(mD->mFrame.get() != image.get())
        mD->mFrame = image->shared_from_this();
    return mD->decode(data, size);
}
