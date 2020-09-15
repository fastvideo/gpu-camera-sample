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
#include "CTPTransport.h"

class TcpClient : public QObject
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
                       AVCodecContext *codec, EncoderType encType, QObject *parent = nullptr);
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
    EncoderType mEncoderType;

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
