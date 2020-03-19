#ifndef TCPCLIENT_H
#define TCPCLIENT_H

#include <QObject>
#include <QThread>
#include <QTcpSocket>
#include <QUdpSocket>
#include <QTimer>

#include <memory>
#include <mutex>

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
#include "ctptransport.h"

class TcpClient : public QObject
{
	Q_OBJECT
public:
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

	explicit TcpClient(QTcpSocket *sock, const QString& url,
					   AVCodecContext *codec, QObject *parent = nullptr);
	~TcpClient();
	/**
	 * @brief sendpkt
	 * send packet to client
	 * @param pkt
	 */
	void sendpkt(AVPacket* pkt);
	/**
	 * @brief isInit
	 * return true if transport ready
	 * @return
	 */
	bool isInit() const;

signals:
	void removeClient(TcpClient *);

public slots:
	void connected();
	void disconnected();
	void readyRead();

private:
	QTcpSocket *m_socket;
	std::unique_ptr< QThread > m_thread;
	QByteArray m_buffer;
	QStringList m_gets;
    bool m_isInit = false;
    bool m_done = false;

    bool m_isCustomTransport = false;
    std::unique_ptr<QUdpSocket> m_udpSocket;
    CTPTransport m_ctpTransport;
    std::vector<QByteArray> m_packets;

	QString m_options;
	QString m_UserAgent;
	QString m_CSeq;
	QString m_Session;
	QString m_url;
    QString m_fmtSdp = "26";

    qint64 m_frameCnt = 0;

	ushort m_clientPort1 = 0;
	ushort m_clientPort2 = 0;
	ushort m_serverPort1 = 6000;
	ushort m_serverPort2 = 6001;

	bool m_isReadBytes = false;
	bool m_isRawHasRead = false;
	int m_rawSize = 0;
	QByteArray m_rawData;
	bool m_isWaitOk = false;

	QString m_transportStr = "RTP/AVP/UDP";
    enum Transport{UDP, TCP, CTP};
	Transport m_transport = UDP;

	int m_state = NONE;

	AVFormatContext *m_fmt = nullptr;
	AVCodec *m_codec = nullptr;
	AVCodecContext *m_ctx_main = nullptr;

    std::mutex m_mutex;

	void parseBuffer();
	void parseLines();

	void sendConnect();
	void sendOk();
	void sendDescribe();
	void sendSetup();
	void sendSetupOk();
    void sendReallChallenge();
    void sendRequiredReply();

	void setPlay();

	void parseTransport(const QString& transport);

	QString generateSDP(ushort portudp = 0);
};

#endif // TCPCLIENT_H
