#ifndef RTSPSTREAMERSERVER_H
#define RTSPSTREAMERSERVER_H

#include <QObject>
#include <QThread>
#include <QTcpServer>
#include <QTcpSocket>
#include <QElapsedTimer>
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
#include "tcpclient.h"

class RTSPStreamerServer : public QObject
{
	Q_OBJECT
public:
	typedef enum
	{
		etNVENC,
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
    /**
     * @brief isStarted
     * @return
     */
    bool isStarted() const;

	/**
	 * @brief addRGBFrame
	 * the function to split rgb frame to tiles
	 * @param rgbPtr
	 * @param linesize - size of one line in bytes
	 * @return
	 */
	bool addBigFrame (unsigned char* rgbPtr, size_t linesize);
	/**
	 * @brief addRGBFrame
	 * default function to add rgb frame
	 * @param rgbPtr
	 * @return
	 */
	bool addFrame (unsigned char* rgbPtr);

	bool startServer();

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
    bool        mMultithreading = true;
    bool        mUseCustomEncodeJpeg = true;
    bool        mUseCustomEncodeH264 = false;

	QElapsedTimer mTimerCtrlFps;
	int mFps = 60;
	int mDelayFps = 1000 / mFps;
	qint64 mCurrentTimeElapsed = 0;

    qint64      mFramesProcessed = 0;
    qint64      mBitrate = 20000000;

    std::unique_ptr<QTcpServer> mServer;
    std::shared_ptr<QThread>    mThread;


    QHostAddress    mHost;
    ushort          mPort;

    AVCodecID       mCodecId = AV_CODEC_ID_MJPEG;
    AVPixelFormat   mPixFmt = AV_PIX_FMT_YUV420P;

    AVCodecContext* mCtx = nullptr;
    AVCodec*        mCodec = nullptr;


    QVector<unsigned char> mEncoderBuffer;
    std::list<TcpClient*>  mClients;

    std::vector<bytearray> mData;
    std::vector<bytearray> mJpegData;

    time_point             mStartTime = time_point (std::chrono::milliseconds(0));

    void doServer();

	void RGB2Yuv420p(unsigned char *destination, unsigned char *rgba, int width, int height);
	void Gray2Yuv420p(unsigned char *destination, unsigned char *rgba, int width, int height);
    void encodeWriteFrame(AVFrame *frame);
    void sendPkt(AVPacket *pkt);

};

#endif // RTSPSTREAMERSERVER_H
