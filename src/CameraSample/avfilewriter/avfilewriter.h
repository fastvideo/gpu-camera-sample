#ifndef AVFILEWRITER_H
#define AVFILEWRITER_H

#include <QObject>
#include <QElapsedTimer>
#include <QScopedPointer>

#include "AsyncFileWriter.h"

#include "common_utils.h"

extern "C" {
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/avstring.h>
}

#include <mutex>

#ifdef __ARM_ARCH
#include "v4l2encoder.h"
#endif

//class TSEncoder;

class AVFileWriter : public AsyncWriter
{
    Q_OBJECT
public:
    typedef enum
    {
        etNVENC,
        etNVENC_HEVC,
    } EncoderType;

    explicit AVFileWriter(QObject *parent = nullptr);
    ~AVFileWriter();

    /**
     * @brief setEncodeNv12Fun
     * @param fun
     */
    void setEncodeNv12Fun(TEncodeFun fun);
    /**
     * @brief setEncodeP010Fun
     * @param fun
     */
    void setEncodeYUV420Fun(TEncodeFun fun);

    bool open(int w, int h, int bitrate, int fps, bool isHEVC, const QString &outFileName);
    void close();

    double duration() const {
        return mDuration;
    }

    void restart(const QString& newFileName);
    QString getLastError(){return mErrStr;}

    int getFrameSizes(fastChannelDescription_t fs[3]);

signals:


    // AsyncWriter interface
protected:
    void processTask(FileWriterTask *task);

private:
    QString     mErrStr;
    bool mIsError = false;
    int mWidth = 0;
    int mHeight = 0;
    int mFps = 0;
    int mBitrate = 20000000;
    float mDelayFps = 0;
    EncoderType mEncoderType = etNVENC;
    QByteArray mEncoderBuffer;
    QElapsedTimer mTimerCtrlFps;
    int64_t mFramesProcessed = 0;
    int mChannels = 3;
    bool mIsInitialized = 0;
    TEncodeFun mNv12Encode;
    TEncodeFun mYUV420Encode;
    QString mFileName;
    QString mCodecName;

    AVCodecID       mCodecId = AV_CODEC_ID_H264;
    AVPixelFormat   mPixFmt = AV_PIX_FMT_YUV420P;

    AVFormatContext* mFmt = nullptr;
    AVStream*       mStream = nullptr;
    AVCodecContext* mCtx = nullptr;
    AVCodec*        mCodec = nullptr;
    AVBufferRef*    mHwDeviceCtx = NULL;

    std::shared_ptr< std::thread > mFrameThread;
//    std::shared_ptr< std::thread > mPacketThread;

#ifdef __ARM_ARCH
    userbuffer mUserBuffer;
    QScopedPointer<v4l2Encoder> mV4L2Encoder;
    void encodeWriteFrame(uint8_t *buf, int width, int height);
#endif

    struct FrameBuffer{
        QByteArray buffer;
        FrameBuffer(){}
        FrameBuffer(uchar *buf, int size){
            Q_UNUSED(buf)
            Q_UNUSED(size)
//            if(size > 0){
//                buffer.resize(size);
//                memcpy(buffer.data(), buf, buffer.size());
//            }
        }
    };
    // very unsafe
    size_t mMaxFrameBuffers = 2;
    std::list<FrameBuffer> mFrameBuffers;
    size_t mMaxPackets = 5;
//    std::list<AVPacket*> mPackets;
//    std::mutex mPacketMutex;
    std::mutex mFrameMutex;
    bool mDone = false;
    double mDuration = 0;

    //QScopedPointer<TSEncoder> mFileWriter;

    void doEncodeFrame();
    bool addInternalFrame();

    //void RGB2Yuv420p(unsigned char *destination, unsigned char *rgba, int width, int height);
    //void Gray2Yuv420p(unsigned char *destination, unsigned char *rgba, int width, int height);
    void encodeWriteFrame(AVFrame *frame, int *pgot = nullptr);
    void sendPkt(AVPacket *pkt);
    //void doPacketWrite();
    void restartAsync();

    void write_pkt(AVPacket* enc_pkt, int stream_index = 0);
};



#endif // AVFILEWRITER_H
