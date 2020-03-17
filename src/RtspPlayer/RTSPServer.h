#ifndef RTSPSERVER_H
#define RTSPSERVER_H

#include <QObject>
#include <QThread>
#include <QHostAddress>
#include <QUdpSocket>
#include <QTcpSocket>
#include <QMap>
#include <QVariant>

#include <queue>
#include <memory>
#include <mutex>
#include <thread>

#include "common.h"
#include "ctptransport.h"
#include "fastvideo_decoder.h"

extern "C"{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

class RTSPServer : public QObject, public AbstractReceiver
{
    Q_OBJECT
public:
    explicit RTSPServer(QObject *parent = nullptr);
    ~RTSPServer();

    QString url() const;
    void startServer(const QString &url, const QMap<QString, QVariant>& additional_params);

    void stopServer();

    void setUseFastVideo(bool val);

    void setUseCustomProtocol(bool val);

    void setMaxWidthFastvideo(uint val);

    bool isFrameExists() const override;
    PImage takeFrame() override;
    uint64_t bytesReaded() override;

    bool isError() const;
    QString errorStr();

    quint32 framesCount() const;

    bool done() const;
    bool isDoStop() const;
    void doProcess();

	QMap<QString, double> durations();

signals:
    void startStopServer(bool);

private:
    std::unique_ptr<std::thread> m_thread;
    std::unique_ptr<std::thread> m_udpThread;
    std::unique_ptr<std::thread> m_decThread;
    bool m_done = false;
    bool m_playing = false;
    QMap<QString, QVariant> m_addiotionalParams;
    QString m_url;
    QString m_error;
    bool m_isClient = false;

	QMap<QString, double> m_durations;

    int m_bufferUdp = 1000000;

    quint32 m_framesCount = 0;
    quint64 m_bytesReaded = 0;

    AVCodec *m_codec = nullptr;
    AVFormatContext *m_fmtctx = nullptr;
    AVCodecContext *m_cdcctx = nullptr;
    AVInputFormat *m_inputfmt = nullptr;
    bool m_is_open = false;
    bool m_doStop = false;

    std::vector< PImage > m_partImages;
    std::vector< bytearray > m_encodedData;
    size_t m_cntX = 0;
    size_t m_cntY = 0;
    size_t m_width = 0;
    size_t m_height = 0;
    bool m_updatedImages = false;

	//std::queue< PImage > m_frames;
	//size_t m_max_frames = 10;
    std::mutex m_mutex;
    std::mutex m_mutexDecoder;

    std::unique_ptr<fastvideo_decoder> m_decoderFv;
    bool m_useFastvideo = false;
    PImage m_fvImage;
	bool m_image_updated = false;

    bool m_useCustomProtocol = false;
    std::unique_ptr<QUdpSocket> m_socket;
    std::unique_ptr<QTcpSocket> m_socketTcp;
    QByteArray m_buffer;
    QStringList m_gets;

    /**
     * state of machine for rtsp server
     */
    enum {NONE,
          CONNECT,
          CONNECTED,
          ANNOUNCE,
          DESCRIBE,
          WAITDESCRIBE,
          OPTIONS,
          SETUP,
          SETUP_OK,
          SET_PARAMETER,
          PLAY,
          PLAYING,
          RealChallenge,
          RealChallenge2,
          PAUSE};

    bool m_isReadBytes = false;
    bool m_isRawHasRead = false;
    int m_rawSize = 0;
    QByteArray m_rawData;
    bool m_isWaitOk = false;
    QString m_UserAgent;
    QString m_Session;
    QString m_CSeq;
    int m_iCSec = 1;
    QString m_options;

    std::queue<QByteArray> m_encodecPkts;
    std::mutex m_mutexDec;
    size_t m_max_buffer_size = 2;

    enum {
        CODEC_JPEG,
        CODEC_H264
    };
    int m_idCodec = CODEC_JPEG;
    ushort m_clientPort1 = 8000;
    ushort m_clientPort2 = 8001;
    CTPTransport m_ctpTransport;

    int m_state = NONE;
    QString m_transportStr;

    enum Transport{UDP, TCP, CTP};
    Transport m_transport = CTP;

    uint32_t m_dropFrames = 0;

    void doServer();
    void closeAV();

    void doServerCustom();
    void parseData();
    void parseLines();
    void parseSdp(const QByteArray &sdp);
    void parseTransport(const QString &transport);
    void sendDescribe();
    void sendOk();
    void sendSetup();
    void sendPlay();

    /**
     * @brief doPlay
     * custom udp reader. assembly m_encodecPkts
     */
    void doPlay();
    /**
     * @brief doDecode
     * decode images from m_encodecPkts
     */
    void doDecode();
    void decode_packet(const QByteArray &enc);
    /**
     * @brief writeToTcpSocket
     * @param lines
     */
    void writeToTcpSocket(const QString& lines);
    /**
     * @brief decode_packet
     * @param pkt
     * @param customDecode
     */
    void decode_packet(AVPacket *pkt, bool customDecode = false);
    void decode_packet(AVPacket *pkt, PImage &image);
    void analyze_frame(AVFrame *frame);
    void waitUntilStopStreaming();
    void getEncodedData(AVPacket *pkt, bytearray& data);

    void getImage(AVFrame *frame, PImage &obj);
    /**
     * @brief assemblyImages
     * if custom headers is exists then copy packet to m_encodedData
     * @param pkt
     * @return
     */
    bool assemblyImages(AVPacket *pkt);
    /**
     * @brief assemplyOutput
     * assembly part of images to one image
     */
    void assemplyOutput();
};

#endif // RTSPSERVER_H
