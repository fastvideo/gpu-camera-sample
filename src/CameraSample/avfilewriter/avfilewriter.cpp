#include "avfilewriter.h"

#include "common_utils.h"
#include "vutils.h"

extern "C"{
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
};

#include <fastvideo_sdk.h>

//////////////////////////////////////////////

#define GOP_SIZE    3;

class TSEncoder{
public:
    std::string mFileName;
    int mFps = 0;

    TSEncoder(){
    }
    ~TSEncoder(){
        try{
            if(stream){
                avcodec_close(stream->codec);
            }
            if(fmt){
                for(uint i = 0; i < fmt->nb_streams; ++i){
                    av_freep(&fmt->streams[i]->codec);
                    av_freep(&fmt->streams[i]);
                }
                avformat_free_context(fmt);
            }
        }catch(...){

        }
    }

    void open(const std::string& fileName, const std::string& format, const std::string &codecName,
              int width, int height, AVPixelFormat pixfmt,
              int fps, int bitrate){
        mFileName = fileName;
        mFps = fps;

        //av_log_set_level(AV_LOG_TRACE);

        AVOutputFormat *ofmt = av_guess_format(format.c_str(), mFileName.c_str(), nullptr);
        if(!ofmt){
            return;
        }
        int res = avformat_alloc_output_context2(&fmt, ofmt, nullptr, mFileName.c_str());
        if(res < 0)
            return;

        fmt->oformat = ofmt;

        res = avio_open(&fmt->pb, mFileName.c_str(), AVIO_FLAG_WRITE);

        codec = avcodec_find_encoder_by_name(codecName.c_str());
        if(!codec)
            return;

        stream = avformat_new_stream(fmt, codec);
        stream->id = fmt->nb_streams - 1;

        stream->time_base = {1, mFps};

        ctx = stream->codec;
        ctx->bit_rate = bitrate;
        ctx->time_base = {1, mFps};         // for test. maybe do not affect
        ctx->framerate = {mFps, 1};         // for test. maybe do not affect
        ctx->width = width;
        ctx->height = height;
        ctx->gop_size = GOP_SIZE;
        if(ctx->codec_id == AV_CODEC_ID_HEVC)
            ctx->max_b_frames = 0;
        else
            ctx->max_b_frames = GOP_SIZE;
        ctx->pix_fmt = pixfmt;

        if (fmt->oformat->flags & AVFMT_GLOBALHEADER)
            ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

        res = avcodec_open2(ctx, codec, nullptr);
        if(res >= 0){
            isInit = true;

//            if(id > 0 && stream_in){
//                open_encoder(id, stream_in);
//            }

            res = avformat_write_header(fmt, nullptr);
        }else{
            char buf[100];
            av_make_error_string(buf, sizeof(buf), res);
            printf("error %s\n", buf);
        }

        timer.restart();
        mTimer.restart();
    }

    int idstreams[10] = {0};

    void write_pkt(const uint8_t* data, size_t size, int stream_index = 0){
        AVPacket pkt;
        av_init_packet(&pkt);

        av_new_packet(&pkt, static_cast<int>(size));
        std::copy(data, data + size, pkt.data);

        //printf("Packet size %d\n", size);

        //pkt.pts = (m_frameNum += 900./mFps);
        pkt.pts = mTimer.elapsed();
        pkt.dts = pkt.pts;
        pkt.stream_index = idstreams[stream_index];

        av_interleaved_write_frame(fmt, &pkt);
        av_packet_unref(&pkt);

        timer.restart();
    }

    void close(){
        if(fmt){
            if(isInit){
                av_write_trailer(fmt);
                avio_close(fmt->pb);
                avformat_flush(fmt);
            }
            printf("close record\n");
            fmt->pb = nullptr;
        }
    }

    bool isInit = false;

private:
    QElapsedTimer mTimer;
    AVCodec *codec = nullptr;
    AVCodecContext *ctx = nullptr;
    AVFormatContext *fmt = nullptr;
    AVStream *stream = nullptr;
    QElapsedTimer timer;
    //uint64_t m_frameNum = 0;
};


//////////////////////////////////////////////

AVFileWriter::AVFileWriter(QObject *parent) : AsyncWriter(-1, parent)
{
    avcodec_register_all();
    av_register_all();
    avformat_network_init();
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

bool AVFileWriter::open(int w, int h, int bitrate, int fps, bool isHEVC)
{
    //av_log_set_level(AV_LOG_TRACE);

    mDone = false;
    int ret = 0;

    mWidth = w;
    mHeight = h;
    mFps = fps;
    mBitrate = bitrate;

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
    }else if(mEncoderType == etNVENC_HEVC){
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

    mCtx = avcodec_alloc_context3(mCodec);

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

    mDelayFps = 1000 / mFps;

    mTimerCtrlFps.restart();

    mEncoderBuffer.resize(mWidth * mHeight * 4);

    mIsInitialized = true;

    return true;
}

void AVFileWriter::close()
{
    mDone = true;
    if(mFileWriter){
        mFileWriter->close();
    }
    mFileWriter.reset();

    if(mFrameThread.get()){
        mFrameThread->join();
        mFrameThread.reset();
    }
    if(mPacketThread.get()){
        mPacketThread->join();
        mPacketThread.reset();
    }

    try{
        if(mCtx)
        {
            avcodec_close(mCtx);
            avcodec_free_context(&mCtx);
        }
        if(mHwDeviceCtx){
            av_buffer_unref(&mHwDeviceCtx);
            mHwDeviceCtx = nullptr;
        }
    }catch(...){
        qDebug("unknown error when close codec and context");
    }
}

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

            addInternalFrame();
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
        frm->pts = mFramesProcessed++;
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
                }
            }
//            if(rgbPtr != nullptr)
//            {
//                RGB2Yuv420p((unsigned char*)mEncoderBuffer.data(), rgbPtr, mWidth, mHeight);
//            }else{
//                return false;
//            }
#endif
        }
#ifdef __ARM_ARCH
        if(mEncoderType == etNVENC || mEncoderType == etNVENC_HEVC){
            encodeWriteFrame(nullptr, mWidth, mHeight);
        }
#else
        {
            encodeWriteFrame(frm);
        }
#endif

        av_frame_free(&frm);
    }

    double duration = getDuration(starttime);
    qDebug("encode duration %f", duration);
    mDuration = duration;

    if(ret == 0)
    {
        mFramesProcessed++;
        return true;
    }

    return false;
}

void AVFileWriter::RGB2Yuv420p(unsigned char *yuv,
                               unsigned char *rgb,
                               int width,
                               int height)
{
    const size_t image_size = width * height;
    unsigned char *dst_y = yuv;
    unsigned char *dst_u = yuv + image_size;
    unsigned char *dst_v = yuv + image_size * 5 / 4;

      // Y plane
      for(size_t i = 0; i < image_size; i++)
      {
          int r = rgb[3 * i];
          int g = rgb[3 * i + 1];
          int b = rgb[3 * i + 2];
          *dst_y++ = ((67316 * r + 132154 * g + 25666 * b) >> 18 ) + 16;
      }

      // U and V plane
      for(size_t y = 0; y < height; y+=2)
      {
          for(size_t x = 0; x < width; x+=2)
          {
              const size_t i = y * width + x;
              int r = rgb[3 * i];
              int g = rgb[3 * i + 1];
              int b = rgb[3 * i + 2];
              *dst_u++ = ((-38856 * r - 76282 * g + 115138 * b ) >> 18 ) + 128;
              *dst_v++ = ((115138 * r - 96414 * g - 18724 * b) >> 18 ) + 128;
          }
      }
}

void AVFileWriter::Gray2Yuv420p(unsigned char *yuv, unsigned char *gray, int width, int height)
{
    const size_t image_size = width * height;
    unsigned char *dst_y = yuv;
    unsigned char *dst_u = yuv + image_size;
    unsigned char *dst_v = yuv + image_size * 5 / 4;

      // Y plane
      for(size_t i = 0; i < image_size; i++)
      {
          int r = gray[i];
          *dst_y++ = ((67316 * r + 132154 * r + 25666 * r) >> 18 ) + 16;
      }

      // U and V plane
      for(size_t y = 0; y < height; y+=2)
      {
          for(size_t x = 0; x < width; x+=2)
          {
              const size_t i = y * width + x;
              int r = gray[i];
              *dst_u++ = ((-38856 * r - 76282 * r + 115138 * r ) >> 18 ) + 128;
              *dst_v++ = ((115138 * r - 96414 * r - 18724 * r) >> 18 ) + 128;
          }
      }
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

void AVFileWriter::encodeWriteFrame(AVFrame *frame)
{
    int ret = 0, got;
    AVPacket enc_pkt;
    enc_pkt.data = nullptr;
    enc_pkt.size = 0;

    av_init_packet(&enc_pkt);

    ret = avcodec_encode_video2(mCtx, &enc_pkt, frame, &got);
    if(got > 0){
        enc_pkt.pts = frame->pts;
        sendPkt(&enc_pkt);
        av_packet_unref(&enc_pkt);
    }
}

void AVFileWriter::sendPkt(AVPacket *pkt)
{
    if(mPackets.size() >= mMaxPackets){
        return;
    }
    std::lock_guard<std::mutex> lg(mPacketMutex);


    AVPacket *pkt_ = av_packet_clone(pkt);
    mPackets.push_back(pkt_);

    if(!mPacketThread.get()){
        mPacketThread.reset(new std::thread([this](){
            doPacketWrite();
        }));
    }
}

void AVFileWriter::doPacketWrite()
{
    while(!mDone){
        if(mPackets.empty()){
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }else{
            mPacketMutex.lock();
            AVPacket* fb = mPackets.front();
            mPackets.pop_front();
            mPacketMutex.unlock();

            if(!mFileWriter){
                mFileWriter.reset(new TSEncoder);
                mFileWriter->open(mFileName.toStdString(), "mkv", mCodecName.toLatin1().data(),
                                  mWidth, mHeight, mPixFmt, mFps, mBitrate);
            }
            if(mFileWriter->isInit){
                mFileWriter->write_pkt((uint8_t*)fb->data, fb->size);
            }

            av_packet_unref(fb);
        }
    }
}



