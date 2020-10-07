/*
 Copyright 2011-2019 Fastvideo, LLC.
 All rights reserved.

 This file is a part of the GPUCameraSample project
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 3. Any third-party SDKs from that project (XIMEA SDK, Fastvideo SDK, etc.) are licensed on different terms. Please see their corresponding license terms.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 The views and conclusions contained in the software and documentation are those
 of the authors and should not be interpreted as representing official policies,
 either expressed or implied, of the FreeBSD Project.
*/

#ifndef RTSPSTREAMERSERVER_H
#define RTSPSTREAMERSERVER_H

#include <QObject>
#include <QThread>
#include <QTcpServer>
#include <QTcpSocket>
#include <QElapsedTimer>
#include <QScopedPointer>
#include <QTimer>
#include <memory>
#include <list>

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

#include "common_utils.h"
#include "TcpClient.h"

#ifdef __ARM_ARCH
#include "v4l2encoder.h"
#endif

class RTSPStreamerServer : public QObject
{
	Q_OBJECT
public:
	typedef enum
	{
		etNVENC,
        etNVENC_HEVC,
		etJPEG,
		etJ2K
	} EncoderType;

    explicit RTSPStreamerServer(int width, int height, int channels, const QString& url,
                                EncoderType encType, unsigned bitrate, QObject *parent = nullptr);
	~RTSPStreamerServer();

    void setBitrate(qint64 bitrate);

    /**
     * @brief setEncodeFun
     * set encoded function
     * @param fun
     */
	void setEncodeFun(TEncodeRgb fun);
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
    /**
     * @brief setMultithreading
     * set use multithreading
     * @param val
     */
    void setMultithreading(bool val);
    /**
     * @brief multithreading
     * @return
     */
    bool multithreading() const;

    void setUseCustomEncodeJpeg(bool val);
    void setUseCustomEncodeH264(bool val);

	bool isError() const;
	QString errorStr() const;
    /**
     * @brief isConnected
     * @return
     */
    bool isConnected() const;
	bool isAnyClientInit() const;
    /**
     * @brief isStarted
     * @return
     */
    bool isStarted() const;

	/**
	 * @brief addRGBFrame
	 * default function to add rgb frame
	 * @param rgbPtr
	 * @return
	 */
	bool addFrame (unsigned char* rgbPtr);

	bool startServer();

    double duration() const;

signals:

public slots:
	void removeClient(TcpClient *client);
	void newConnection();

private:

    bool        mIsError = false;
    int         mWidth = 0;
    int         mHeight = 0;
    int         mChannels = 0;
    QString     mUrl;
    bool        mIsInitialized = false;
    QString     mErrStr;
    EncoderType mEncoderType = etNVENC;
    TEncodeRgb  mJpegEncode;
    TEncodeFun mNv12Encode;
    TEncodeFun mYUV420Encode;
    bool        mMultithreading = true;
    bool        mUseCustomEncodeJpeg = true;
    bool        mUseCustomEncodeH264 = false;
    double      mDuration = 0;

	QElapsedTimer mTimerCtrlFps;
	int mFps = 60;
	int mDelayFps = 1000 / mFps;
	qint64 mCurrentTimeElapsed = 0;

    qint64      mFramesProcessed = 0;
    qint64      mBitrate = 20000000;

    std::unique_ptr<QTcpServer> mServer;
    std::shared_ptr<QThread>    mThread;

	std::shared_ptr< std::thread > mFrameThread;

#ifdef __ARM_ARCH
    userbuffer mUserBuffer;
    QScopedPointer<v4l2Encoder> mV4L2Encoder;
    void encodeWriteFrame(uint8_t *buf, int width, int height);
#endif

	struct FrameBuffer{
		uchar *buffer = nullptr;
		size_t size = 0;
		FrameBuffer(){}
		FrameBuffer(uchar *buf){ buffer = buf; }
	};
	// very unsafe
	size_t mMaxFrameBuffers = 2;
	std::list<FrameBuffer> mFrameBuffers;
	std::mutex mFrameMutex;
	bool mDone = false;
	void doFrameBuffer();
	bool addInternalFrame(uchar *rgbPtr);

    QHostAddress    mHost;
    ushort          mPort;

    AVCodecID       mCodecId = AV_CODEC_ID_MJPEG;
    AVPixelFormat   mPixFmt = AV_PIX_FMT_YUV420P;

    AVCodecContext* mCtx = nullptr;
    AVCodec*        mCodec = nullptr;
    AVBufferRef*    mHwDeviceCtx = NULL;

    QVector<unsigned char> mEncoderBuffer;
    std::list<TcpClient*>  mClients;

    std::vector<bytearray> mData;
	std::vector<Buffer> mJpegData;

    time_point             mStartTime = time_point (std::chrono::milliseconds(0));

    void doServer();

	void RGB2Yuv420p(unsigned char *destination, unsigned char *rgba, int width, int height);
	void Gray2Yuv420p(unsigned char *destination, unsigned char *rgba, int width, int height);
    void encodeWriteFrame(AVFrame *frame);
    void sendPkt(AVPacket *pkt);

};

#endif // RTSPSTREAMERSERVER_H
