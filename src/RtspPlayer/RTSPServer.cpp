#include "RTSPServer.h"

#include <QTimer>
#include <QTime>
#include <QFile>
#include <QPoint>

#include <thread>
#include <chrono>

#ifdef _MSC_VER
#include <WinSock2.h>
#pragma comment(lib, "WS2_32.lib")
#else
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netinet/in.h>
#include <unistd.h>
#include <errno.h>

#define SOCKET_ERROR (-1)
#define closesocket(x) close((int)x)
#endif

#include "GLImageViewer.h"
#include "vdecoder.h"
#include "common_utils.h"
#include "jpegenc.h"

#ifndef __ARM_ARCH
#include "cuviddecoder.h"
#else
#endif

////////////////////////////
void copyRect(const PImage &part, size_t xOff, size_t yOff, PImage &out);

////////////////////////////

bool extractAddress(const QString &_url, QHostAddress& _addr, ushort &_port)
{
    QString addr, port, url = _url;
    if(url.indexOf("rtsp://") >= 0){
        url = url.remove(0, 7);
        int pos = url.indexOf(":", 0);
        if(pos >= 0){
            addr = url.left(pos);

            url = url.remove(0, pos + 1);
            pos = url.indexOf("/");
            if(pos >= 0){
                port = url.left(pos);
            }else{
                port = url;
            }
            _addr = QHostAddress(addr);
            _port = port.toUInt();

            if(!_port){
                qDebug("wrong url. port");
                return false;
            }

            return true;
        }else{
            qDebug("wrong url. port");
            return false;
        }
    }else{
        qDebug("wrong url. name");
        return false;
    }
}

////////////////////////////

RTSPServer::RTSPServer(GLRenderer *renderer, QObject *parent)
	: QObject(parent)
	, mRenderer(renderer)
{
    mVDecoder.reset(new VDecoder);

    m_url = "rtsp://127.0.0.1:1234/live.sdp";

#ifdef _MSC_VER
    WSADATA WSAData;
    WORD wV = MAKEWORD(2, 2);
    WSAStartup(wV, &WSAData);
#endif
}

RTSPServer::~RTSPServer()
{
	mRenderer = nullptr;

    qDebug("close server..");

    stopServer();

    qDebug("server closed");
}

QString RTSPServer::url() const
{
	return m_url;
}

bool RTSPServer::isServerOpened() const
{
	return m_isServerOpened;
}

void RTSPServer::startServer(const QString &url, const QMap<QString, QVariant> &additional_params)
{
    if(m_playing){
        return;
    }
    m_done = false;
    m_dropFrames = 0;
    m_bytesReaded = 0;
	m_addiotionalParams = additional_params;

	if(additional_params.contains("ctp")){
		setUseCustomProtocol(additional_params["ctp"].toBool());
	}
	if(additional_params.contains("mjpeg_fastvideo")){
		setUseFastVideo(additional_params["mjpeg_fastvideo"].toBool());
	}
    if(additional_params.contains("h264")){
        int id = additional_params["h264"].toInt();
        if(id == 1){
			setH264Codec("h264_cuvid");
        }else if(id == 2){
            setH264Codec("h264");
        }else if(id == 3){
            setH264Codec("nv");
        }
    }
	if(additional_params.contains("client")){
		m_isClient = true;
	}
	if(additional_params.contains("buffer")){
		m_bufferUdp = m_addiotionalParams["buffer"].toInt();
	}

	m_url = url;
	m_isServerOpened = true;
	m_playing = true;
	m_isStartDecode = true;

    m_timerStartServer.start();

    m_thread.reset(new std::thread([this](){
        if(m_useCustomProtocol){
            doServerCustom();
        }else{
            doServer();
        }
    }));

}

void RTSPServer::stopServer()
{
    if(mHSocket){
        closesocket(mHSocket);
        mHSocket = 0;
    }

    m_done = true;

    if(mVDecoder.get())
        mVDecoder->waitUntilStopStreaming();

    mVDecoder.reset();

    while(m_playing){
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if(m_decThread.get()){
        m_decThread->join();
        m_decThread.reset();
    }

    if(m_thread.get()){
        m_thread->join();
        m_thread.reset();
    }

    if(m_udpThread.get()){
        m_udpThread->join();
        m_udpThread.reset();
    }

    //m_socket.reset();
    m_socketTcp.reset();

    closeAV();

    m_state = NONE;

//    while(!m_frames.empty())
//        m_frames.pop();
}

void RTSPServer::setUseFastVideo(bool val)
{
	std::lock_guard<std::mutex> lg(m_mutexDecoder);
    if(mVDecoder.get()){
        mVDecoder->setUseFastVideo(val);
    }
	m_fvImage.reset();
}

void RTSPServer::setUseCustomProtocol(bool val)
{
	m_useCustomProtocol = val;
}

void RTSPServer::setH264Codec(const QString &codec)
{
    if(mVDecoder.get())
        mVDecoder->setH264Codec(codec);
}

bool RTSPServer::isCuvidFound() const
{
    return mVDecoder.get() && mVDecoder->isCuvidFound();
}

bool RTSPServer::isMJpeg() const
{
    return mVDecoder.get() && mVDecoder->isMJpeg();
}

bool RTSPServer::isFrameExists() const
{
	return m_fvImage.get() && m_image_updated;// m_useFastvideo && m_fvImage.get() || !m_frames.empty();
}

uint64_t RTSPServer::bytesReaded()
{
    return m_bytesReaded;
}

bool RTSPServer::isError() const
{
    return !m_error.isEmpty();
}

QString RTSPServer::errorStr()
{
    return m_error;
}

quint32 RTSPServer::framesCount() const
{
	return m_framesCount;
}

void RTSPServer::startDecode()
{
	m_isStartDecode = true;
}

void RTSPServer::stopDecode()
{
	m_isStartDecode = false;
}

QMap<QString, double> RTSPServer::durations()
{
    mMutexDurs.lock();
    QMap<QString, double> durs = m_durations;
    mMutexDurs.unlock();
    return durs;
}

bool RTSPServer::done() const
{
    return m_done;
}

bool RTSPServer::isLive() const
{
    return m_timerStartServer.elapsed() < max_server_waiting_ms;
}

void RTSPServer::doServer()
{
    int res = 0;
    m_bytesReaded = 0;

    if(mVDecoder.get() == nullptr)
        return;

    if(!mVDecoder->initContext(m_url, m_isClient)){
        m_playing = false;
        return;
    }

    emit startStopServer(true);

    do{
        if(res >= 0){
            qDebug("<< Server opened >>");

            res = mVDecoder->initStream();

            if(res >= 0){
                qDebug("Decoder opened");
                m_clientStarted = true;

                for(;!m_done;){
                    res = mVDecoder->readPacket();

                    if(res >= 0){
						/// try to check packet to additionaly header
						/// if true then move to assemply packets
						/// else simply decode
//						if(!assemblyImages(&pkt))
                        decodePacket();
                    }else{
                        char buf[100];
                        av_make_error_string(buf, sizeof(buf), res);
                        m_error = QString("read frame error: %1 (%2)").arg(res).arg(buf);
                        qDebug("error: %s", buf);
                        break;
                    }

                    mVDecoder->freePacket();
                }
            }else{
                m_error = QString("stream not found: %1").arg(res);
                break;
            }

        }else{
            m_error = QString("input not opened: %1").arg(res);
            m_done = true;
        }
    }while(0);

    emit startStopServer(false);

    m_error = "closed correctly";

    m_playing = false;
    m_done = true;
	m_isServerOpened = false;

    qDebug("Decode ended");
}

void RTSPServer::closeAV()
{
    if(mVDecoder.get())
        mVDecoder->close();
}

void RTSPServer::decodePacket()
{
    m_timerStartServer.restart();

    if(!m_isStartDecode || !mVDecoder.get())
		return;

    double duration = -1;

    QString str;
    if(mVDecoder->decodePacket(m_fvImage, str, m_bytesReaded, duration)){
        if(!str.isEmpty()){
            mMutexDurs.lock();
            m_durations[str] = duration;
            mMutexDurs.unlock();
        }

        updateRenderer();
    }
}

void RTSPServer::updateRenderer()
{
    m_framesCount++;
	if(mRenderer){
		mRenderer->loadImage(m_fvImage, m_bytesReaded);
		mRenderer->update();
	}
}

void RTSPServer::doServerCustom()
{
    if(!m_isClient){
        m_error = "Work only as client";
        qDebug("Work only as client");
        return;
    }

    m_socketTcp.reset(new QTcpSocket);

    QHostAddress addr;
    ushort port;
    if(!extractAddress(m_url, addr, port)){
        return;
    }

    m_socketTcp->connectToHost(addr, port);

    if(!m_socketTcp->isOpen()){
        m_error = "Socket not open";
        return;
    }

    m_state = OPTIONS;
    QString request = "OPTIONS " + m_url + " RTSP/1.0\r\n"
                                           "\r\n";
    writeToTcpSocket(request);

    while(!m_done && m_socketTcp->isOpen()){
        if(!m_socketTcp.get())
            break;
        bool res = m_socketTcp->waitForReadyRead(3000);
        if(res && m_socketTcp.get()){
            QByteArray ba = m_socketTcp->read(2048 * 1024);
            if(ba.isEmpty()){
				std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }else{
                m_buffer.append(ba);
                parseData();
            }
        }
    }
    m_socketTcp->abort();
    m_socketTcp.reset();

    m_playing = false;
	m_isServerOpened = false;
}

void RTSPServer::parseData()
{
    while(!m_buffer.isEmpty()){
        if(m_isReadBytes){
            if(m_buffer.size() >= m_rawSize){
                m_rawData = m_buffer.left(m_rawSize);
                m_buffer.remove(0, m_rawSize);
                m_isReadBytes = false;
                m_isRawHasRead = true;
                parseSdp(m_rawData);
            }else{
                break;
            }
            continue;
        }
        int pos = m_buffer.indexOf("\r\n\r\n");
        if(pos >= 0){
            //qDebug("%s", m_buffer.data());
            QByteArray lines = m_buffer.left(pos);
            m_buffer = m_buffer.remove(0, pos + 4);

            QList<QByteArray> ll = lines.split('\n');
            foreach (QByteArray l, ll) {
                m_gets.push_back(l.trimmed());
            }

            qDebug("----- begin ------ ");
            parseLines();
            qDebug("----- end ------ ");
        }else{
            break;
        }
    }
}

void RTSPServer::parseLines()
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
                if(m_state == OPTIONS){
                    m_state = DESCRIBE;
                }
            }else if(cmd == "Transport:"){
                parseTransport(option);
            }else if(cmd == "SETUP"){
            }else if(cmd == "PLAY"){
            }else if(cmd == "OPTIONS"){
            }else if(cmd == "ClientChallenge:"){
            }if(cmd == "Require:"){
            }if(cmd == "SET_PARAMETER"){
            }else if(cmd == "User-Agent:"){
            }else if(cmd == "CSeq:"){
            }else if(cmd == "Content-Length:"){
                m_rawSize = option.toInt();
                if(m_rawSize){
                    m_rawData.clear();
                    m_isReadBytes = true;
                }
                m_state = CONNECTED;
            }else if(cmd == "DESCRIBE"){
            }
            qDebug("command: %s option: %s;  state %d", cmd.toStdString().c_str(), option.toStdString().c_str(), m_state);
        }
    }

    switch (m_state) {
    case DESCRIBE:
        sendDescribe();
        break;
    case SETUP:
        sendPlay();
        break;
    case CONNECTED:

        break;
    }
}

void RTSPServer::parseSdp(const QByteArray &sdp)
{
    if(!mVDecoder.get())
        return;
    QStringList sl = QString(sdp).split("\n");

    bool search_codec = false;
    for(QString s: sl){
        int pos = s.indexOf("m=video");
        if(pos >= 0){
            pos = s.indexOf("RTP/AVP");
            if(pos >= 0){
                QString fmt = s.right(s.size() - pos - QString("RTP/AVP").size()).trimmed();
                if(fmt == "26"){
                    mVDecoder->setCodec(VDecoder::CODEC_JPEG);         /// select jpeg codec
                }else if(fmt == "96"){
                    search_codec = true;
                }
            }
            if(!search_codec){
                m_state = SETUP;
                sendSetup();
                break;
            }
        }
        if(search_codec){
            pos = s.indexOf("a=rtpmap");
            if(pos >= 0){
                pos = s.toUpper().indexOf("H265");
                if(pos >= 0){
                    mVDecoder->setCodec(VDecoder::CODEC_HEVC);         /// select h265 codec
                }else{
                    pos = s.toUpper().indexOf("H264");
                    if(pos >= 0){
                        mVDecoder->setCodec(VDecoder::CODEC_H264);         /// select h264 codec
                    }
                }
                m_state = SETUP;
                sendSetup();
            }
        }
    }
}

void RTSPServer::parseTransport(const QString &transport)
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
                    m_transport = UDP;                  /// select of udp transport
                    m_transportStr = "RTP/AVP/UDP";
                }
                if(s.indexOf("TCP") >= 0){
                    m_transport = TCP;                  /// select of tcp transport. not realized
                    m_transportStr = "RTP/AVP/TCP";
                }
                if(s.indexOf("CTP") >= 0){
                    m_transport = CTP;                  /// select of ctp transport (custom transport)
                }
            }
        }
    }
}

void RTSPServer::writeToTcpSocket(const QString &lines)
{
    QByteArray data = lines.toLatin1();
    m_socketTcp->write(data);
    m_socketTcp->waitForBytesWritten();
}

void RTSPServer::sendDescribe()
{
    m_CSeq = QString::number(m_iCSec);
    m_UserAgent = "Custom";
    QString reply =
            "DESCRIBE " + m_url + " RTSP/1.0\r\n"
            "Accept: application/sdp\r\n"
            "CSeq: " + m_CSeq + "\r\n"
            "User-Agent: " + m_UserAgent + "\r\n"
            "\r\n";
    writeToTcpSocket(reply);
}

void RTSPServer::sendOk()
{
    QString reply = "RTSP/1.0 200 OK\r\n"
                    "Server: Custom\r\n"
                    "CSeq: " + m_CSeq + "\r\n"
                    "\r\n";
    writeToTcpSocket(reply);
}

void RTSPServer::sendSetup()
{
    QString request = "SETUP " + m_url + " RTSP/1.0\r\n" +
            QString("Transport: RTP/AVP/CTP;unicast;client_port=%1-%2;\r\n")
            .arg(m_clientPort1).arg(m_clientPort2) +
            "CSeq: " + m_CSeq + "\r\n"
            "User-agent: " + m_UserAgent + "\r\n"
            "\r\n";
    writeToTcpSocket(request);
}

void RTSPServer::sendPlay()
{
    if(mVDecoder.get() == nullptr || !mVDecoder->initDecoder()){
        return;
    }

    m_udpThread.reset(new std::thread([this](){
        doPlay();
    }));

    m_decThread.reset(new std::thread([this](){
        doDecode();
    }));

    QString request = "PLAY " + m_url + " RTSP/1.0\r\n"
            "Range: npt=0.000-\r\n"
            "CSeq: " + m_CSeq + "\r\n"
            "User-agent: " + m_UserAgent + "\r\n"
            "\r\n";
    writeToTcpSocket(request);

    m_clientStarted = true;

    m_state = PLAYING;
}

void RTSPServer::doPlay()
{
//    m_socket.reset(new QUdpSocket);
//    m_socket->moveToThread(QThread::currentThread());
//    m_socket->bind(m_clientPort1);
    if(mHSocket){
        closesocket(mHSocket);
        mHSocket = 0;
    }

    mHSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if(!mHSocket){
        qDebug("Error create socket");
        return;
    }

    const int timeout_in_mseconds = 10;
#ifdef _MSC_VER
    // WINDOWS
    DWORD timeout = timeout_in_mseconds;
    setsockopt(mHSocket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof timeout);
#else
    // LINUX
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = timeout_in_mseconds * 1000;
    setsockopt(mHSocket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

#endif

    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr("0.0.0.0");
    addr.sin_port = htons(m_clientPort1);
    int res = bind(mHSocket, (sockaddr*)&addr, sizeof(addr));
    if(res == SOCKET_ERROR){
        qDebug("Socket bind error");
        return;
    }


    int opt = m_bufferUdp;
    //setsockopt(m_socket->socketDescriptor(), SOL_SOCKET, SO_RCVBUF, (char*)&opt, sizeof(opt));
    setsockopt(mHSocket, SOL_SOCKET, SO_RCVBUF, (char*)&opt, sizeof(opt));

    emit startStopServer(true);

    uchar data[65536] = {0};
    while(!m_done){
        res = recv(mHSocket, (char*)data, sizeof(data), 0);
        if(res > 0){
            m_ctpTransport.addUdpPacket(data, res);

            if(m_ctpTransport.isPacketAssembly()){
                if(m_encodecPkts.size() < m_max_buffer_size){
                    m_mutexDec.lock();
                    QByteArray data = m_ctpTransport.getPacket();
                    m_encodecPkts.push(data);
                    m_mutexDec.unlock();

                    mMutexDurs.lock();
					m_durations = mergeMaps(m_durations, m_ctpTransport.durations());
                    mMutexDurs.unlock();

                    m_bytesReaded += data.size();
                }else{
                    qDebug("packet cannot show. overflow buffer. drop frames %d", ++m_dropFrames);
                }

                m_ctpTransport.clearPacket();
            }

        }else{
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    emit startStopServer(false);

    if(mHSocket){
        closesocket(mHSocket);
        mHSocket = 0;
    }
    //m_socket.reset();
}

void RTSPServer::doDecode()
{
    while(!m_done){
        if(m_encodecPkts.empty()){
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }else{
            m_mutexDec.lock();
            QByteArray enc = m_encodecPkts.front();
            m_encodecPkts.pop();
            m_mutexDec.unlock();

            decode_packet(enc);
        }
    }
}

void RTSPServer::decode_packet(const QByteArray &enc)
{
    m_timerStartServer.restart();
    if(!m_isStartDecode)
		return;
	//PImage obj;

	std::lock_guard<std::mutex> lg(m_mutexDecoder);

    double duration = -1;

	QString name;

    if(mVDecoder->decodePacket(enc, m_fvImage, name, duration)){
        if(!name.isEmpty()){
            mMutexDurs.lock();
            m_durations[name] = duration;
            mMutexDurs.unlock();
        }

        updateRenderer();

        qDebug("decode duration(ms): %f         \r", duration);
    }

//    if(!m_useFastvideo){
//        if(obj.get()){
//            m_mutex.lock();
//            m_frames.push(obj);
//            m_mutex.unlock();
//        }
//    }
}
