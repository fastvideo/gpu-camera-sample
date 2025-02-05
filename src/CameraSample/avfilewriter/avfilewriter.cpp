#include "avfilewriter.h"

#include "common_utils.h"
#include "vutils.h"

#include <QFileInfo>

extern "C"{
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
};

#include <fastvideo_sdk.h>

//////////////////////////////////////////////

#define GOP_SIZE    3;

//////////////////////////////////////////////

AVFileWriter::AVFileWriter(QObject *parent) : AsyncWriter(-1, parent)
{
    avcodec_register_all();
    av_register_all();
    avformat_network_init();
    //av_log_set_level(AV_LOG_TRACE);
}


AVFileWriter::~AVFileWriter()
{
    close();
}

void AVFileWriter::setEncodeNv12Fun(TEncodeFun fun)
{
    mNv12Encode = fun;
}

void AVFileWriter::setEncodeYUV420Fun(TEncodeFun fun)
{
    mYUV420Encode = fun;
}

#pragma warning(push)
#pragma warning(disable : 4996)

bool AVFileWriter::open(int w, int h, int bitrate, int fps, bool isHEVC, const QString& outFileName)
{
    //av_log_set_level(AV_LOG_TRACE);

    mDone = false;
    int ret = 0;

    mWidth = w;
    mHeight = h;
    mFps = fps;
    mBitrate = bitrate;
    mFileName = outFileName;

    mEncoderType = isHEVC? etNVENC_HEVC : etNVENC;

    if(mEncoderType == etNVENC)
    {
#ifdef __ARM_ARCH
        mV4L2Encoder.reset(new v4l2Encoder());
        mV4L2Encoder->setIDRInterval(1);
        mV4L2Encoder->setEnableAllIFrameEncode(true);
        mV4L2Encoder->setInsertSpsPpsAtIdrEnabled(true);
        mV4L2Encoder->setInsertVuiEnabled(true);
        mV4L2Encoder->setIFrameInterval(3);
        mV4L2Encoder->setNumBFrames(3);
        mPixFmt = AV_PIX_FMT_YUV420P;
//        mCodec = avcodec_find_encoder_by_name("h264_v4l2m2m");
#else
        mCodec = avcodec_find_encoder_by_name("h264_nvenc");
        mPixFmt = AV_PIX_FMT_NV12;
#endif
        if(!mCodec)
        {
            mCodec = avcodec_find_encoder_by_name("libx264");
            if(!mCodec)
            {
                mIsError = true;
                mErrStr = "Codec not found";
                return false;
            }
        }
    }
    else if(mEncoderType == etNVENC_HEVC)
    {
#ifdef __ARM_ARCH
        mV4L2Encoder.reset(new v4l2Encoder());
        mV4L2Encoder->setEncoder(v4l2Encoder::eHEVC);
        mV4L2Encoder->setIDRInterval(1);
        mV4L2Encoder->setEnableAllIFrameEncode(true);
        mV4L2Encoder->setInsertSpsPpsAtIdrEnabled(true);
        mV4L2Encoder->setInsertVuiEnabled(true);
        mV4L2Encoder->setIFrameInterval(1);
        mPixFmt = AV_PIX_FMT_YUV420P;
//        mCodec = avcodec_find_encoder_by_name("h264_v4l2m2m");
#else
        mCodec = avcodec_find_encoder_by_name("hevc_nvenc");
        mPixFmt = AV_PIX_FMT_P010;
#endif
        if(!mCodec)
        {
            mCodec = avcodec_find_encoder_by_name("libx265");
            if(!mCodec)
            {
                mCodec = avcodec_find_encoder(AV_CODEC_ID_HEVC);
                if(!mCodec){
                    mIsError = true;
                    mErrStr = "Codec not found";
                    return false;
                }
            }
        }
    }

    mCodecName = mCodec->name;
    mCodecId = mCodec->id;

    std::string format = "mp4";

    AVOutputFormat *ofmt = av_guess_format(format.c_str(), mFileName.toStdString().c_str(), nullptr);
    if(!ofmt)
    {
        return false;
    }
    int res = avformat_alloc_output_context2(&mFmt, ofmt, nullptr, mFileName.toStdString().c_str());
    if(res < 0)
        return false;

    mFmt->oformat = ofmt;

    res = avio_open(&mFmt->pb, mFileName.toStdString().c_str(), AVIO_FLAG_WRITE);

    mStream = avformat_new_stream(mFmt, mCodec);
    mStream->id = mFmt->nb_streams - 1;

    mStream->time_base = {1, mFps};

    mCtx = mStream->codec;

    //mCtx = avcodec_alloc_context3(mCodec);

    mCtx->bit_rate = mBitrate;

    {
        mCtx->width = mWidth;
        mCtx->height = mHeight;
    }

    //frames per second
    mCtx->time_base = {1, mFps};         // for test. maybe do not affect
    mCtx->framerate = {mFps, 1};         // for test. maybe do not affect
    mCtx->gop_size = GOP_SIZE;
    mCtx->max_b_frames = GOP_SIZE;
#ifdef __ARM_ARCH
    mCtx->pix_fmt = mPixFmt;
#else
    mCtx->pix_fmt = AV_PIX_FMT_CUDA;

    avcodec_parameters_from_context(mStream->codecpar, mCtx);

    ret = av_hwdevice_ctx_create(&mHwDeviceCtx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);

    ret = set_hwframe_ctx(mCtx, mHwDeviceCtx, mWidth, mHeight, mPixFmt);

    if(ret < 0){
        return false;
    }
#endif

    if(mEncoderType == etNVENC || mEncoderType == etNVENC_HEVC)
    {
        mCtx->max_b_frames = 0;        // codec do not open for mjpeg
        mCtx->keyint_min = 0;
        mCtx->flags |= AV_CODEC_FLAG_LOW_DELAY;
        mCtx->flags2 |= AV_CODEC_FLAG2_FAST;
    }

    AVDictionary *dict = nullptr;
    av_dict_set(&dict, "c", "v", 0);

    if(mCodecId == AV_CODEC_ID_H264 || mCodecId == AV_CODEC_ID_HEVC || mCodecId == AV_CODEC_ID_INDEO3)
    {
        av_dict_set(&dict, "zerolatency", "1", 0);
        //av_dict_set(&dict, "preset", "fast", 0);
        av_dict_set(&dict, "movflags", "+faststart", 0);
        av_dict_set(&dict, "delay", "0", 0);
        av_dict_set(&dict, "rc", "cbr_ld_hq", 0);
    }

    ret = avcodec_open2(mCtx, mCodec, &dict);
    if(ret < 0)
    {
        char buf[100];
        av_make_error_string(buf, sizeof(buf), ret);
        mErrStr = QString(QStringLiteral("avcodec_open2 failed, code: %1 (%2)")).arg(ret, 0, 16).arg(buf);
        mIsError = true;
        return false;
    }

    res = avformat_write_header(mFmt, nullptr);

    mDelayFps = 1000 / mFps;

    mTimerCtrlFps.restart();

    mEncoderBuffer.resize(mWidth * mHeight * 4);

    mIsInitialized = true;
    mProcessed = 0;
    mDropped = 0;

    return true;
}

void AVFileWriter::close()
{
    std::lock_guard<std::mutex> lg(mFrameMutex);

    mDone = true;
    if(mFrameThread.get())
    {
        mFrameThread->join();
        mFrameThread.reset();
    }

    try
    {
        if(mFmt){
            int got = 0;
            do{
                got = 0;
                encodeWriteFrame(NULL, &got);
                qDebug("write frame %d", got);
            }while(got > 0);

            av_write_trailer(mFmt);
        }
        if(mHwDeviceCtx)
        {
            av_buffer_unref(&mHwDeviceCtx);
            mHwDeviceCtx = nullptr;
        }
        if(mCtx)
        {
            avcodec_close(mCtx);
            avio_close(mFmt->pb);
            mFmt->pb = nullptr;
            /* free the streams */
            for(uint i = 0; i < mFmt->nb_streams; i++) {
                avcodec_free_context(&mFmt->streams[i]->codec);
                mFmt->streams[i]->codec = nullptr;
                av_freep(&mFmt->streams[i]);
            }
            mCtx = nullptr;
        }
        if(mFmt)
        {
            avformat_free_context(mFmt);
            qDebug("close record");
            mStream = nullptr;
            mFmt = nullptr;
        }
    }
    catch(...)
    {
        qDebug("unknown error when close codec and context");
    }
    mIsInitialized = false;
}

#pragma warning(pop)

void AVFileWriter::processTask(FileWriterTask *task)
{
    if(task == nullptr || !mIsInitialized)
        return;

    std::lock_guard<std::mutex> lg(mFrameMutex);
    if(mFileName.isEmpty())
        mFileName = task->fileName;

    if(mFrameBuffers.size() < mMaxFrameBuffers)
        mFrameBuffers.push_back(FrameBuffer(task->data, task->size));

    if(!mFrameThread.get()){
        mFrameThread.reset(new std::thread([this](){
            QThread::currentThread()->setObjectName("FrameThread");
            doEncodeFrame();
        }));
    }
}

void AVFileWriter::doEncodeFrame()
{
    while(!mDone){
        if(mFrameBuffers.empty()){
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }else{
            mFrameMutex.lock();
            //FrameBuffer fb = mFrameBuffers.front();
            mFrameBuffers.pop_front();
            mFrameMutex.unlock();

            while(!mDone && addInternalFrame()){};
        }
    }
}

bool AVFileWriter::addInternalFrame()
{
    auto starttime = getNow();

    if(!mIsInitialized)
        return false;
    int ret = 0;

    if(((mCodecId == AV_CODEC_ID_H264 || mCodecId == AV_CODEC_ID_INDEO3 || mCodecId == AV_CODEC_ID_HEVC)))
    {
        AVFrame* frm = av_frame_alloc();
        frm->width = mWidth;
        frm->height = mHeight;
        frm->format = mPixFmt;
        frm->pts = mFramesProcessed;
        //Set frame->data pointers manually
        if(mChannels == 1/* && rgbPtr != nullptr*/)
        {
            // did not supported yet
            //Gray2Yuv420p((unsigned char*)mEncoderBuffer.data(), rgbPtr, mWidth, mHeight);
        }
        else
        {
#ifdef __ARM_ARCH
             {
                uint8_t *data[3] = {nullptr, nullptr, nullptr};
                int lines[3] = {0, 0, 0};
                if(mV4L2Encoder->getInputBuffers3(data, lines, mWidth, mHeight)){
                    fastChannelDescription_t fs[3];
                    fs[0].data = (unsigned char*)data[0];
                    fs[0].pitch = lines[0];
                    fs[0].height = mHeight;
                    fs[0].width = mWidth;

                    fs[1].data = (unsigned char*)data[1];
                    fs[1].pitch = lines[1];
                    fs[1].height = mHeight/2;
                    fs[1].width = mWidth;

                    fs[2].data = (unsigned char*)data[2];
                    fs[2].pitch = lines[2];
                    fs[2].height = mHeight/2;
                    fs[2].width = mWidth;

                     mYUV420Encode((unsigned char*)&fs, 8);
                     ret = fs[0].pitch > 0? 0 : -1;

                     mV4L2Encoder->putInputBuffers3();
                }
            }
#else
             ret = av_hwframe_get_buffer(mCtx->hw_frames_ctx, frm, 0);

             {
                if(mNv12Encode != nullptr && mCodecId == AV_CODEC_ID_H264){
                    frm->format = AV_PIX_FMT_NV12;

                    fastChannelDescription_t fs[3];
                    fs[0].data = frm->data[0];
                    fs[0].pitch = frm->linesize[0];
                    fs[0].height = mHeight;
                    fs[0].width = mWidth;

                    fs[1].data = frm->data[1];
                    fs[1].pitch = frm->linesize[1];
                    fs[1].height = mHeight/2;
                    fs[1].width = mWidth;

                    fs[2].data = frm->data[2];
                    fs[2].pitch = frm->linesize[2];
                    fs[2].height = mHeight;
                    fs[2].width = mWidth;

                    mNv12Encode((unsigned char*)&fs, 8);
                    ret = fs[0].pitch > 0? 0 : -1;
                }
                else if(mYUV420Encode != nullptr && mCodecId == AV_CODEC_ID_HEVC){
                    fastChannelDescription_t fs[3];
                    fs[0].data = frm->data[0];
                    fs[0].pitch = frm->linesize[0];
                    fs[0].height = mHeight;
                    fs[0].width = mWidth;

                    fs[1].data = frm->data[1];
                    fs[1].pitch = frm->linesize[1];
                    fs[1].height = mHeight/2;
                    fs[1].width = mWidth;

                    fs[2].data = frm->data[2];
                    fs[2].pitch = frm->linesize[2];
                    fs[2].height = mHeight/2;
                    fs[2].width = mWidth;

                    mYUV420Encode((unsigned char*)&fs, 10);
                    ret = fs[0].pitch > 0? 0 : -1;
                }
            }
#endif
        }
        if(ret >= 0){
#ifdef __ARM_ARCH
            if(mEncoderType == etNVENC || mEncoderType == etNVENC_HEVC){
                encodeWriteFrame(nullptr, mWidth, mHeight);
            }
#else
            {
                encodeWriteFrame(frm);
            }
#endif
        }

        av_frame_free(&frm);
    }

    double duration = getDuration(starttime);
    //qDebug("encode duration %f", duration);
    mDuration = duration;

    if(ret == 0)
    {
        mFramesProcessed += 1000/mFps;
        return true;
    }

    return ret >= 0;
}

#ifdef __ARM_ARCH
void AVFileWriter::encodeWriteFrame(uint8_t *buf, int width, int height)
{
    if(mV4L2Encoder.data()){
        if(mV4L2Encoder->getEncodedData(mUserBuffer)){
            if(!mUserBuffer.empty()){
                AVPacket enc_pkt;
                enc_pkt.data = nullptr;
                enc_pkt.size = 0;
                av_init_packet(&enc_pkt);

                av_new_packet(&enc_pkt, static_cast<int>(mUserBuffer.size()));
                enc_pkt.pts = enc_pkt.dts = mFramesProcessed;
                enc_pkt.flags = AV_PKT_FLAG_KEY;
                std::copy(mUserBuffer.data(),mUserBuffer.data() + mUserBuffer.size(), enc_pkt.data);

                sendPkt(&enc_pkt);
                av_packet_unref(&enc_pkt);
            }
        }
    }
}
#endif

void AVFileWriter::encodeWriteFrame(AVFrame *frame, int* pgot)
{
    int ret = 0;
    AVPacket enc_pkt;
    enc_pkt.data = nullptr;
    enc_pkt.size = 0;

    av_init_packet(&enc_pkt);

    ret = avcodec_send_frame(mCtx, frame);
    while(ret >= 0){
        ret = avcodec_receive_packet(mCtx, &enc_pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        enc_pkt.pts = frame->pts;
        if(pgot){
            *pgot = 1;
        }
        sendPkt(&enc_pkt);
        av_packet_unref(&enc_pkt);
    }
}

void AVFileWriter::sendPkt(AVPacket *pkt)
{
    write_pkt(pkt);
}

void AVFileWriter::restart(const QString& /*newFileName*/)
{
    //qDebug("Restarting...");
    wake();
}

int AVFileWriter::getFrameSizes(fastChannelDescription_t fs[])
{
    int ret;
#ifdef __ARM_ARCH
    {
        uint8_t *data[3] = {nullptr, nullptr, nullptr};
        int lines[3] = {0, 0, 0};
        if(mV4L2Encoder->getInputBuffers3(data, lines, mWidth, mHeight)){
            fs[0].data = nullptr;
            fs[0].pitch = lines[0];
            fs[0].height = mHeight;
            fs[0].width = mWidth;

            fs[1].data = nullptr;
            fs[1].pitch = lines[1];
            fs[1].height = mHeight/2;
            fs[1].width = mWidth;

            fs[2].data = nullptr;
            fs[2].pitch = lines[2];
            fs[2].height = mHeight/2;
            fs[2].width = mWidth;
        }
    }
#else
    AVFrame* frm = av_frame_alloc();
    ret = av_hwframe_get_buffer(mCtx->hw_frames_ctx, frm, 0);

    {
        if(mNv12Encode != nullptr && mCodecId == AV_CODEC_ID_H264){
            frm->format = AV_PIX_FMT_NV12;

            //fastChannelDescription_t fs[3];
            fs[0].data = nullptr;
            fs[0].pitch = frm->linesize[0];
            fs[0].height = mHeight;
            fs[0].width = mWidth;

            fs[1].data = nullptr;
            fs[1].pitch = frm->linesize[1];
            fs[1].height = mHeight/2;
            fs[1].width = mWidth;

            fs[2].data = nullptr;
            fs[2].pitch = frm->linesize[2];
            fs[2].height = mHeight;
            fs[2].width = mWidth;
        }
        else if(mYUV420Encode != nullptr && mCodecId == AV_CODEC_ID_HEVC){
            //fastChannelDescription_t fs[3];
            fs[0].data = nullptr;
            fs[0].pitch = frm->linesize[0];
            fs[0].height = mHeight;
            fs[0].width = mWidth;

            fs[1].data = nullptr;
            fs[1].pitch = frm->linesize[1];
            fs[1].height = mHeight/2;
            fs[1].width = mWidth;

            fs[2].data = nullptr;
            fs[2].pitch = frm->linesize[2];
            fs[2].height = mHeight/2;
            fs[2].width = mWidth;
        }
    }
    av_frame_free(&frm);
#endif
    return ret;
}

void AVFileWriter::restartAsync()
{

}

void AVFileWriter::write_pkt(AVPacket *enc_pkt, int )
{
    av_interleaved_write_frame(mFmt, enc_pkt);
}
