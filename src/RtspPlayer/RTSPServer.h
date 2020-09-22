#ifndef RTSPSERVER_H
#define RTSPSERVER_H

#include <QObject>
#include <QThread>
#include <QHostAddress>
#include <QUdpSocket>
#include <QTcpSocket>
#include <QMap>
#include <QVariant>
#include <QElapsedTimer>

#include <queue>
#include <memory>
#include <mutex>
#include <thread>

#include "common.h"
#include "CTPTransport.h"

class VDecoder;
class GLRenderer;

class RTSPServer : public QObject, public AbstractReceiver
{
    Q_OBJECT
public:
	explicit RTSPServer(GLRenderer* renderer, QObject *parent = nullptr);
    ~RTSPServer();

    QString url() const;

	bool isServerOpened() const;
    void startServer(const QString &url, const QMap<QString, QVariant>& additional_params);
    void stopServer();

    bool isClientStarted() const { return m_clientStarted;}

    void setUseFastVideo(bool val);

    void setUseCustomProtocol(bool val);

	void setH264Codec(const QString& codec);

    void setMaxWidthFastvideo(uint val);

	bool isCuvidFound() const;
	bool isMJpeg() const;

	bool isFrameExists() const;
    uint64_t bytesReaded() override;

    bool isError() const;
    QString errorStr();

    quint32 framesCount() const;

	void startDecode();
	void stopDecode();

    bool done() const;
    bool isDoStop() const;

    bool isLive() const;

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
	bool m_isServerOpened = false;
    bool m_clientStarted = false;

	QMap<QString, double> m_durations;

	int m_bufferUdp = 5000000;

    quint32 m_framesCount = 0;
    quint64 m_bytesReaded = 0;

	bool m_isStartDecode = true;

	//std::queue< PImage > m_frames;
	//size_t m_max_frames = 10;
    std::mutex m_mutex;
    std::mutex m_mutexDecoder;
    std::mutex mMutexDurs;

    std::unique_ptr<VDecoder> mVDecoder;
    PImage m_fvImage;
	bool m_image_updated = false;

    bool m_useCustomProtocol = false;

    QElapsedTimer m_timerStartServer;
    //std::unique_ptr<QUdpSocket> m_socket;
#ifdef _MSC_VER
    uint64_t mHSocket = 0;
#else
    int mHSocket = 0;
#endif
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

    ushort m_clientPort1 = 8000;
    ushort m_clientPort2 = 8001;
    CTPTransport m_ctpTransport;

    int m_state = NONE;
    QString m_transportStr;

    enum Transport{UDP, TCP, CTP};
    Transport m_transport = CTP;

    uint32_t m_dropFrames = 0;

	GLRenderer* mRenderer = nullptr;

    qint64 max_server_waiting_ms = 20000;

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
    void decodePacket();
    /**
     * @brief writeToTcpSocket
     * @param lines
     */
    void writeToTcpSocket(const QString& lines);

	void updateRenderer();
//    /**
//     * @brief assemblyImages
//     * if custom headers is exists then copy packet to m_encodedData
//     * @param pkt
//     * @return
//     */
//    bool assemblyImages(AVPacket *pkt);
//    /**
//     * @brief assemplyOutput
//     * assembly part of images to one image
//     */
//    void assemplyOutput();
};

#endif // RTSPSERVER_H
