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

#include "TcpClient.h"

#include <QDateTime>
#include <QCryptographicHash>
#include <QHostAddress>
#include <QList>
#include <QByteArray>

#ifdef _MSC_VER
#include <WinSock2.h>
#pragma comment(lib, "WS2_32.lib")
#else
#include <sys/socket.h>
#endif

#define RTSP_DEFAULT_PORT   554
#define RTSPS_DEFAULT_PORT  322
#define RTSP_MAX_TRANSPORTS 8
#define RTSP_TCP_MAX_PACKET_SIZE 1472
#define RTSP_DEFAULT_NB_AUDIO_CHANNELS 1
#define RTSP_DEFAULT_AUDIO_SAMPLERATE 44100
#define RTSP_RTP_PORT_MIN 5000
#define RTSP_RTP_PORT_MAX 65000

TcpClient::TcpClient(QTcpSocket *sock, const QString &url, AVCodecContext *codec, EncoderType encType, QObject *parent)
	: QObject(parent)
	, m_socket(sock)
	, m_url(url)
	, m_ctx_main(codec)
    , mEncoderType(encType)
{
    if(m_ctx_main && m_ctx_main->codec){
        m_codec = avcodec_find_encoder_by_name(m_ctx_main->codec->name);
        if(m_codec && (mEncoderType == etNVENC || mEncoderType == etNVENC_HEVC)){
            m_fmtSdp = "96";
        }
    }
//	m_thread.reset(new QThread);
//	m_thread->setObjectName("client");
//	moveToThread(m_thread.get());
//	m_thread->start();

//	sock->moveToThread(m_thread.get());

    m_serverPort1 = (rand() % 55000) + 5000;
    m_serverPort2 = m_serverPort1 + 1;

	connect(m_socket, SIGNAL(connected()), this, SLOT(connected()));
	connect(m_socket, SIGNAL(disconnected()), this, SLOT(disconnected()));
	connect(m_socket, SIGNAL(readyRead()), this, SLOT(readyRead()));
}

TcpClient::~TcpClient()
{
    m_done = true;

	if(m_thread.get()){
		m_thread->quit();
		m_thread->wait();
	}

    if(m_udpSocket.get()){
        m_udpSocket->abort();
        m_udpSocket.reset();
    }

	if(m_fmt){
		avformat_free_context(m_fmt);
	}
}

void TcpClient::sendpkt(AVPacket *pkt)
{
	std::lock_guard<std::mutex> lg(m_mutex);

	auto starttime = getNow();

    if(m_isCustomTransport && m_udpSocket.get()){
        m_ctpTransport.createPacket(pkt->data, pkt->size, m_packets);
        for(QByteArray& d: m_packets){
            m_udpSocket->writeDatagram(d, m_socket->peerAddress(), m_clientPort1);
        }
    }else if(m_fmt && m_isInit){
        pkt->stream_index = 0;

        //AVStream *st = m_fmt->streams[pkt->stream_index];

        //AVRational *time_base = &m_fmt->streams[pkt->stream_index]->time_base;

        pkt->pts = pkt->pts * 90000/60;     /// 60 fps

        int ret;
        //ret = avformat_write_header(m_fmt, nullptr);
        ret = av_write_frame(m_fmt, pkt);
	}

	double duration = getDuration(starttime);
	qDebug("send duration %f", duration);
}

bool TcpClient::isInit() const
{
	return m_isInit;
}

void TcpClient::connected()
{

}

void TcpClient::disconnected()
{
    m_mutex.lock();
    m_isInit = false;
    m_mutex.unlock();

    if(m_fmt){
        //ret = avio_open(&m_fmt->pb, m_fmt->filename, AVIO_FLAG_WRITE);
        avio_close(m_fmt->pb);
        avcodec_close(m_fmt->streams[0]->codec);
        avformat_free_context(m_fmt);
        m_fmt = nullptr;
    }

	emit removeClient(this);
}

void TcpClient::readyRead()
{
	int size = 0;
	do{
		QByteArray ba = m_socket->read(8192);
		size = ba.size();

		m_buffer.append(ba);
		parseBuffer();
	}while(size > 0);
}

void TcpClient::parseBuffer()
{
	while(!m_buffer.isEmpty()){
		if(m_isReadBytes){
			if(m_buffer.size() >= m_rawSize){
				m_rawData = m_buffer.left(m_rawSize);
				m_buffer.remove(0, m_rawSize);
				m_isReadBytes = false;
				m_isRawHasRead = true;
				//parseSdp(m_rawData);
			}else{
				break;
			}
			continue;
		}
			int pos = m_buffer.indexOf("\r\n\r\n");
			if(pos >= 0){
                //qDebug("%s\n", m_buffer.data());
				QByteArray lines = m_buffer.left(pos);
                m_buffer = m_buffer.remove(0, pos + 4);

				QList<QByteArray> ll = lines.split('\n');
				foreach (QByteArray l, ll) {
					m_gets.push_back(l.trimmed());
				}

                qDebug("----- begin ------ \n");
                parseLines();
                qDebug("----- end ------ \n");
			}else{
				break;
		}
	}
}


void TcpClient::parseLines()
{
	while(!m_gets.empty()){
		QString line = m_gets.front();
		m_gets.pop_front();

		line = line.trimmed();
		if(line.isEmpty())
			continue;

		int pos = line.indexOf(" ");

		if(pos >= 0){
			QString cmd = line.left(pos);
			QString option = line.remove(0, pos + 1);

			if(cmd == "RTSP/1.0"){
				int status = option.left(option.indexOf(" ")).toInt();
				if(status == 200 && m_state == WAITDESCRIBE)
					m_state = SETUP;
			}else if(cmd == "Transport:"){
				parseTransport(option);
			}else if(cmd == "SETUP"){
				m_state = SETUP_OK;
            }else if(cmd == "PLAY"){
                m_state = PLAY;
            }else if(cmd == "OPTIONS"){
				m_options = option;
				if(m_state == NONE)
					m_state = CONNECT;
            }else if(cmd == "ClientChallenge:"){
                m_state = RealChallenge;
                m_Session = option;
            }if(cmd == "Require:"){
                m_state = RealChallenge2;
            }if(cmd == "SET_PARAMETER"){
                m_state = SET_PARAMETER;
            }else if(cmd == "User-Agent:"){
				m_UserAgent = option;
			}else if(cmd == "CSeq:"){
				m_CSeq = option;
			}else if(cmd == "Content-Length:"){
				m_rawSize = option.toInt();
				if(m_rawSize){
					m_rawData.clear();
					m_isReadBytes = true;
				}
			}else if(cmd == "DESCRIBE"){
				if(m_state == CONNECTED)
					m_state = DESCRIBE;
			}
            qDebug("command: %s option: %s;  state %d\n", cmd.toStdString().c_str(), option.toStdString().c_str(), m_state);
		}
	}

	switch (m_state) {
	case CONNECT:
		sendConnect();
		m_state = CONNECTED;
		break;
	case SETUP:
		m_state = PLAY;
		break;
	case SETUP_OK:
		m_state = PLAY;
		sendSetupOk();
		break;
    case PLAY:
        sendOk();
        setPlay();
        m_state = PLAYING;
        break;
    case RealChallenge:
        sendReallChallenge();
        m_state = RealChallenge2;
        break;
    case RealChallenge2:
        sendRequiredReply();
        m_state = PAUSE;
        break;
    case SET_PARAMETER:
        sendOk();
        m_state = PAUSE;
        break;
	case DESCRIBE:
		sendDescribe();
		m_state = WAITDESCRIBE;
	default:
		break;
	}
}

void TcpClient::sendConnect()
{
	QString reply = "RTSP/1.0 200 OK\r\n"
					"CSeq: " + m_CSeq + "\r\n"
                    + QString("Server: %1\r\n").arg(m_UserAgent) +
                    "Public: ANNOUNCE, PAUSE, SETUP, TEARDOWN, RECORD\r\n"
					"\r\n";

	QByteArray data = reply.toLatin1();
	m_socket->write(data.data(), data.size());
	m_socket->waitForBytesWritten();

//    QString sdp = generateSDP();
//    QByteArray datasdp = sdp.toLatin1();
//    int len = datasdp.size();

//    reply = "ANNOUNCE " + m_url + " RTSP/1.0\r\n"
//            "Content-Type: application/sdp\r\n"
//            "CSeq: " + m_CSeq + "\r\n"
//            + QString("User-Agent: %1\r\n").arg("Custom") +
//            "Content-Length: " + QString::number(len) +
//            "\r\n\r\n";

//    data = reply.toLatin1();
//    m_socket->write(data.data(), data.size());
//    m_socket->waitForBytesWritten();

//    m_socket->write(datasdp.data(), datasdp.size());
//    m_socket->waitForBytesWritten();
}

void TcpClient::sendOk()
{
	QString reply = "RTSP/1.0 200 OK\r\n"
                    "Server: Custom\r\n"
                    "CSeq: " + m_CSeq + "\r\n"
					"\r\n";
	QByteArray data = reply.toLatin1();
	m_socket->write(data.data(), data.size());
	m_socket->waitForBytesWritten();
}

void TcpClient::sendDescribe()
{
	qint64 t = QDateTime::currentMSecsSinceEpoch();
	m_Session = QString::number(t);

	QString ip = m_socket->localAddress().toString();
    //ushort port = m_socket->localPort();

	QString sdp = generateSDP();
	QByteArray datasdp = sdp.toLatin1();
	int len = datasdp.size();

	QString reply =
			"RTSP/1.0 200 OK\r\n"
			"CSeq: " + m_CSeq + "\r\n"
			"Content-Length: " + QString::number(len) + "\r\n"
			"\r\n";
	QByteArray data = reply.toLatin1();
	m_socket->write(data.data(), data.size());
    m_socket->waitForBytesWritten();
    m_socket->write(datasdp.data(), datasdp.size());
	m_socket->waitForBytesWritten();
	m_isWaitOk = false;
}

void TcpClient::sendSetup()
{
	QString ip = m_socket->localAddress().toString();
    //ushort port = m_socket->localPort();

	QString reply =
			QString("Transport: %1;unicast;client_port=5000-20000;mode=record\r\n").arg(m_transportStr) +
			"CSeq: " + m_CSeq + "\r\n"
            "User-Agent: " + "Custom" + "\r\n\r\n";
	QByteArray data = reply.toLatin1();
	m_socket->write(data.data(), data.size());
	m_socket->waitForBytesWritten();
}

void TcpClient::sendSetupOk()
{
	if(!m_clientPort1){
		m_clientPort1 = 5000 + (rand() % 60000);
		m_clientPort2 = m_clientPort1 + 1;
	}

	QString reply =
			"RTSP/1.0 200 OK\r\n"
			"CSeq: " + m_CSeq + "\r\n"
            "Server: " + "Custom" + "\r\n"
            "Transport: " + QString("%5;unicast;mode=receive;client_port=%1-%2;server_port=%3-%4;\r\n")
            .arg(m_clientPort1).arg(m_clientPort2).arg(m_serverPort1).arg(m_serverPort2).arg(m_transportStr) +
			"Session: " + m_Session + "\r\n"
			"\r\n";

	QByteArray data = reply.toLatin1();
	m_socket->write(data.data(), data.size());
    m_socket->waitForBytesWritten();
}

void TcpClient::sendReallChallenge()
{
	if(!m_clientPort1){
		m_clientPort1 = 5000 + (rand() % 60000);
		m_clientPort2 = m_clientPort1 + 1;
	}

    QString reply =
            "RTSP/1.0 200 OK\r\n"
            "RealChallenge1: " + m_Session + "\r\n"
            "ETag: " + m_Session + "\r\n" +
            QString("Transport: %3;client_port=%1;server_port=%2\r\n").arg(m_clientPort1).arg(m_serverPort1).arg(m_transportStr) +
            //"Content-Length: " + QString::number(len) + "\r\n"
            "\r\n";
    QByteArray data = reply.toLatin1();
    m_socket->write(data.data(), data.size());
    m_socket->waitForBytesWritten();
}

void TcpClient::sendRequiredReply()
{
     QString sdp = generateSDP();
    QByteArray datasdp = sdp.toLatin1();
    int len = datasdp.size();

    QString ip = m_socket->localAddress().toString();
    QString reply =
            "RTSP/1.0 200 OK\r\n"
            "ETag: " + m_Session + "\r\n"
            "CSeq: " + m_CSeq + "\r\n"
            "Date: Tue, 10 Oct 2006 13:46:05 GMT\r\n"
            "Session: " + m_Session + "\r\n"
            "x-real-usestrackid: 2\r\n"
            "WWW-Authenticate: RN5 realm=" + ip + ", nonce=\"1160487965929341\"\r\n"
            "Content-Length: " + QString::number(len) + "\r\n"
            "\r\n";
    QByteArray data = reply.toLatin1();
    m_socket->write(data.data(), data.size());
    m_socket->waitForBytesWritten();

    m_socket->write(datasdp.data(), datasdp.size());
    m_socket->waitForBytesWritten();
}

void TcpClient::setPlay()
{
    QString addr = m_socket->peerAddress().toString();
    if(addr.isEmpty())
        addr = "127.0.0.1";
    QString url = QString("rtp://%1:%2?localport=%3").arg(addr).arg(m_clientPort1).arg(m_serverPort1);

    if(m_isCustomTransport){
        m_udpSocket.reset(new QUdpSocket);
//        m_udpThread.reset(new QThread);
//        m_udpThread->setObjectName("udp_thread");
//        m_udpThread->moveToThread(m_udpThread.get());
//        m_udpThread->start();

//        m_udpSocket->moveToThread(m_udpThread.get());
        m_udpSocket->bind(m_serverPort1);

        int opt = buffersize_udp;
        setsockopt(m_udpSocket->socketDescriptor(), SOL_SOCKET, SO_SNDBUF, (char*)&opt, sizeof(opt));

        m_mutex.lock();
        m_isInit = true;
        m_mutex.unlock();
    }else{

        AVOutputFormat *videoFmt = av_guess_format("rtp", url.toLocal8Bit().data(), nullptr);

        if(!videoFmt)
            return;

        int ret = avformat_alloc_output_context2(&m_fmt, videoFmt, nullptr, nullptr);

        av_strlcpy(m_fmt->filename, url.toLocal8Bit().data(), sizeof(m_fmt->filename));

        if(ret < 0){
            char buf[100];
            av_make_error_string(buf, sizeof(buf), ret);
            qDebug("error: %s", buf);
            return;
        }

        ret = avio_open(&m_fmt->pb, m_fmt->filename, AVIO_FLAG_WRITE);

        AVStream* stream = avformat_new_stream(m_fmt, m_codec);
        if(stream){
            stream->id = m_fmt->nb_streams - 1;
        }

        stream->codecpar->bit_rate = m_ctx_main->bit_rate;
        stream->codecpar->width = m_ctx_main->width;
        stream->codecpar->height = m_ctx_main->height;
        stream->codecpar->codec_id = m_codec->id;
        stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
        stream->codecpar->format = m_ctx_main->pix_fmt;
        stream->codec->flags = m_ctx_main->flags;
        stream->codec->flags2 = m_ctx_main->flags2;
        stream->time_base = m_ctx_main->time_base;

    //    ret = avcodec_open2(c, m_codec, nullptr);
    //    if(ret < 0){
    //        return;
    //    }

        AVDictionary *opt = nullptr;
        ret = avformat_write_header(m_fmt, &opt);
        if(ret < 0){
            char buf[100];
            av_make_error_string(buf, sizeof(buf), ret);
            qDebug("error: %s", buf);
		}else{
			m_mutex.lock();
			m_isInit = true;
			m_mutex.unlock();
        }
    }
}

void TcpClient::parseTransport(const QString &transport)
{
	QStringList sl = transport.split(';');
	for(QString &s: sl){
		QStringList sl2 = s.split('=');
		if(sl2.size() > 1){
			if(sl2[0] == "client_port"){
				QStringList sl3 = sl2[1].split('-');
				m_clientPort1 = sl3[0].toUInt();
				m_clientPort2 = sl3[1].toUInt();
			}
		}else{
			if(s.indexOf("AVP") >= 0){
				if(s.indexOf("UDP") >= 0){
					m_transport = UDP;
					m_transportStr = "RTP/AVP/UDP";
				}
				if(s.indexOf("TCP") >= 0){
					m_transport = TCP;
					m_transportStr = "RTP/AVP/TCP";
				}
                if(s.indexOf("CTP") >= 0){
                    m_transport = CTP;
                    m_transportStr = "RTP/AVP/CTP";
                    m_isCustomTransport = true;
                }
			}
		}
	}
}

QString TcpClient::generateSDP(ushort portudp)
{
    QString ip = m_socket->localAddress().toString();
    //ushort port = m_socket->localPort();

    if(!m_codec && mEncoderType != etJPEG)
        return "";

    if(mEncoderType == etJPEG){
        QString sdp =
                QString(
                "v=0\r\n"
                "o=- 0 0 IN IP4 %1\r\n"
                "s=No Name\r\n"
                "c=IN IP4 %1\r\n"
                "t=0 0\r\n"
                "a=tool:libavformat 57.83.100\r\n"
                "m=video %2 RTP/AVP 26\r\n"
                "b=AS:200\r\n"
                "a=control:streamid=0").arg(ip).arg(portudp);
        return sdp;
    }else if(mEncoderType == etNVENC){
        QString sdp = QString(
                    "v=0\r\n"
                    "o=- 0 0 IN IP4 %1\r\n"
                    "s=No Name\r\n"
                    "c=IN IP4 %1\r\n"
                    "t=0 0\r\n"
                    "a=tool:libavformat 58.29.100\r\n"
                    "m=video %2 RTP/AVP 96\r\n"
                    "a=rtpmap:96 H264/90000\r\n"
                    "a=fmtp:96 packetization-mode=1; sprop-parameter-sets=Z2QAHqzZQLQnsBEAAZdPAExLQA8WLZY=,aOvjyyLA; profile-level-id=64001E\r\n"
                    "a=control:streamid=0\r\n").arg(ip).arg(portudp);
        return sdp;
    }else if(mEncoderType == etNVENC_HEVC){
        QString sdp = QString(
                    "v=0\r\n"
                    "o=- 0 0 IN IP4 %1\r\n"
                    "s=No Name\r\n"
                    "c=IN IP4 %1\r\n"
                    "t=0 0\r\n"
                    "a=tool:libavformat 58.29.100\r\n"
                    "m=video %2 RTP/AVP 96\r\n"
                    "a=rtpmap:96 H265/90000\r\n").arg(ip).arg(portudp);
        return sdp;
    }
    return "";
}
