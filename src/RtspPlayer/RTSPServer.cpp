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
#endif

#include "GLImageViewer.h"

#include "common_utils.h"

#include "jpegenc.h"

const int STREAM_TYPE_VIDEO     = 0;
const int MAXIMUM_WAIT          = 3000;

////////////////////////////
void copyRect(const PImage &part, size_t xOff, size_t yOff, PImage &out);


int callback_layer(void* data)
{
    RTSPServer *obj = static_cast<RTSPServer*>(data);
    obj->doProcess();

    if(obj->done() || obj->isDoStop()){
        return 1;
    }

    return 0;
}

////////////////////////////

bool extratAddress(const QString &_url, QHostAddress& _addr, ushort &_port)
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
    av_register_all();
    avformat_network_init();

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
	if(additional_params.contains("h264_cuvid")){
		if(additional_params["h264_cuvid"].toBool()){
			setH264Codec("h264_cuvid");
		}else{
			if(additional_params["libx264"].toBool()){
				setH264Codec("libx264");
			}else{
				setH264Codec("h264");
			}
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

    waitUntilStopStreaming();

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

	m_decoderFv.reset();

    closeAV();

    m_state = NONE;

//    while(!m_frames.empty())
//        m_frames.pop();

    m_partImages.clear();
    m_encodedData.clear();

    m_is_open = false;
}

void RTSPServer::setUseFastVideo(bool val)
{
	std::lock_guard<std::mutex> lg(m_mutexDecoder);
    m_useFastvideo = val;
	m_fvImage.reset();
}

void RTSPServer::setUseCustomProtocol(bool val)
{
	m_useCustomProtocol = val;
}

void RTSPServer::setH264Codec(const QString &codec)
{
	m_codecH264 = codec;
}

bool RTSPServer::isCuvidFound() const
{
	return m_codec && QString(m_codec->name) == "h264_cuvid";
}

bool RTSPServer::isMJpeg() const
{
	return m_idCodec == CODEC_JPEG;
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

bool RTSPServer::isDoStop() const
{
    return m_doStop;
}

void RTSPServer::doProcess()
{

}

QMap<QString, double> RTSPServer::durations()
{
	return m_durations;
}

bool RTSPServer::done() const
{
    return m_done;
}

void RTSPServer::waitUntilStopStreaming()
{
    if(m_is_open){
        QTime time;
        time.start();
        m_doStop = true;
		if(!m_useCustomProtocol){
			while(m_is_open && time.elapsed() < MAXIMUM_WAIT){
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
			if(time.elapsed() > MAXIMUM_WAIT){
				qDebug("Oops. Streaming not stopped");
			}
		}
        m_doStop = false;
        m_is_open = false;

        if(m_cdcctx && avcodec_is_open(m_cdcctx)){
            avcodec_close(m_cdcctx);
        }

        if(m_fmtctx){
            avformat_close_input(&m_fmtctx);
        }
        m_fmtctx = nullptr;
        m_cdcctx = nullptr;
    }
}

void RTSPServer::getEncodedData(AVPacket *pkt, bytearray &data)
{
    data.resize(pkt->size);
    std::copy(pkt->data, pkt->data + pkt->size, data.data());
}

void RTSPServer::doServer()
{
    int res = 0;
    m_fmtctx = avformat_alloc_context();
    m_bytesReaded = 0;

    AVDictionary *dict = nullptr;

    if(!m_isClient){
        av_dict_set(&dict, "rtsp_flags", "listen", 0);
        qDebug("Try to open server %s ..", m_url.toLatin1().data());
    }else{
        qDebug("Try to open address %s ..", m_url.toLatin1().data());
    }

    av_dict_set(&dict, "preset", "slow", 0);
    /// try to decrease latency
    av_dict_set(&dict, "tune", "zerolatency", 0);
    /// need if recevie many packets or packet is big
    av_dict_set(&dict, "buffer_size", "500000", 0);
    /// dont know. copy from originaly ffplay
    av_dict_set(&dict, "scan_all_pmts", "1", AV_DICT_DONT_OVERWRITE);

    try {
        m_is_open = true;

        m_fmtctx->interrupt_callback.callback = callback_layer;
        m_fmtctx->interrupt_callback.opaque = this;

        res = avformat_open_input(&m_fmtctx, m_url.toLatin1().data(), m_inputfmt, &dict);

        if(res < 0){
            qDebug("url %s can'not open", m_url.toLatin1().data());
            m_error = "url can'not open";
            m_is_open = false;
        }
    } catch (...) {
        m_is_open = false;
        m_error = "closed uncorrectrly";
        qDebug("closed input uncorrectly");
        return;
    }

    if(!m_is_open){
        m_playing = false;
        return;
    }

    emit startStopServer(true);

    do{
        if(res >= 0){
            qDebug("<< Server opened >>");

            res = av_find_best_stream(m_fmtctx, AVMEDIA_TYPE_VIDEO, STREAM_TYPE_VIDEO, -1, nullptr, 0);

            if(res >= 0){
                res = avformat_find_stream_info(m_fmtctx, nullptr);

                m_cdcctx = avcodec_alloc_context3(m_codec);
                res = avcodec_parameters_to_context(m_cdcctx, m_fmtctx->streams[STREAM_TYPE_VIDEO]->codecpar);
                if(res < 0){
                    m_error = QString("error copy paramters to context %1").arg(res);
                    m_is_open = false;
                    break;
                }
				m_cdcctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
				m_cdcctx->flags2 |= AV_CODEC_FLAG2_FAST;

                m_codec = avcodec_find_decoder(m_cdcctx->codec_id);
                if(!m_codec){
                    m_error = "Codec not found";
                    m_is_open = false;
                    break;
                }
                if(m_codec->id == AV_CODEC_ID_H264){
					AVCodec *c = avcodec_find_decoder_by_name(m_codecH264.toLatin1().data());
                    if(c) m_codec = c;
                    m_idCodec = CODEC_H264;
                }

                qDebug("Decoder found: %s", m_codec->name);

                AVDictionary *dict = nullptr;
                av_dict_set(&dict, "zerolatency", "1", 0);
                av_dict_set_int(&dict, "buffer_size", m_bufferUdp, 0);

                res = avcodec_open2(m_cdcctx, m_codec, &dict);
                if(res < 0){
                    m_error = QString("decoder not open: %1").arg(res);
                    m_is_open = false;
                    break;
                }
                qDebug("Decoder opened");
                m_clientStarted = true;

                for(;!m_done;){
                    AVPacket pkt;
                    av_init_packet(&pkt);
                    res = av_read_frame(m_fmtctx, &pkt);

                    if(res >= 0){
						/// try to check packet to additionaly header
						/// if true then move to assemply packets
						/// else simply decode
//						if(!assemblyImages(&pkt))
						decode_packet(&pkt);

                        m_bytesReaded += pkt.size;
                    }else{
                        char buf[100];
                        av_make_error_string(buf, sizeof(buf), res);
                        m_error = QString("read frame error: %1 (%2)").arg(res).arg(buf);
                        qDebug("error: %s", buf);
                        break;
                    }

                    av_packet_unref(&pkt);
                }
            }else{
                m_error = QString("stream not found: %1").arg(res);
                m_is_open = false;
                break;
            }

        }else{
            m_error = QString("input not opened: %1").arg(res);
            m_done = true;
            m_is_open = false;
        }
    }while(0);

    emit startStopServer(false);

    m_error = "closed correctly";

    m_playing = false;
    m_is_open = false;
    m_done = true;
	m_isServerOpened = false;

    qDebug("Decode ended");
}

void RTSPServer::closeAV()
{
    if(m_fmtctx){
        avformat_close_input(&m_fmtctx);
        m_fmtctx = nullptr;
    }
    if(m_cdcctx){
        avcodec_free_context(&m_cdcctx);
        m_cdcctx = nullptr;
    }
    m_codec = nullptr;
    m_inputfmt = nullptr;
    m_doStop = false;
}

#pragma warning(push)
#pragma warning(disable : 4189)

void RTSPServer::decode_packet(AVPacket *pkt)
{
	if(!m_isStartDecode)
		return;
	if(m_idCodec != CODEC_JPEG){
        int ret = 0;
		int got = 0;

		auto starttime = getNow();

        AVFrame *frame = av_frame_alloc();

		ret = avcodec_decode_video2(m_cdcctx, frame, &got, pkt);

		if(got > 0){
			analyze_frame(frame, m_fvImage);
            av_frame_unref(frame);

			QString name = m_cdcctx->codec->name;
			QString out = "decode (" + name + "):";

			m_durations[out] = getDuration(starttime);

			updateRenderer();
        }

        av_frame_free(&frame);
    }else{
//        PImage obj;
		auto starttime = getNow();

		if(m_encodedData.empty()){
            m_encodedData.resize(1);
        }
        if(m_partImages.empty()){
            m_partImages.resize(1);
        }
        getEncodedData(pkt, m_encodedData[0]);

		std::lock_guard<std::mutex> lg(m_mutexDecoder);
		if(m_useFastvideo){
			if(!m_decoderFv.get())
				m_decoderFv.reset(new fastvideo_decoder);
			m_decoderFv->decode(m_encodedData[0], m_fvImage, true);
			m_image_updated = true;
        }else{
			jpegenc dec;
			dec.decode(m_encodedData[0], m_partImages[0]);
			if(!m_fvImage.get())
				m_fvImage.reset(new Image);
			m_fvImage->setRGB(m_partImages[0]->width, m_partImages[0]->height);
			copyRect(m_partImages[0], 0, 0, m_fvImage);
			m_image_updated = true;
        }

		QString name = m_cdcctx->codec->name;
		QString out = "decode (" + name + "):";

		m_durations[out] = getDuration(starttime);

		updateRenderer();

//        if(obj.get()){
//            m_mutex.lock();
//			m_frames.push(obj);
//            m_mutex.unlock();
//        }
    }
}

void RTSPServer::decode_packet(AVPacket *pkt, PImage &image)
{
	if(!m_isStartDecode)
		return;

	auto starttime = getNow();

	int got = 0;
    AVFrame *frame = av_frame_alloc();

	/// this function deprecated but comfortable to calculate duration
	int res = avcodec_decode_video2(m_cdcctx, frame, &got, pkt);

	static int cnt = 1;
	static auto time = getNow();

	double duration = 0;
	if(cnt == 1){
		time = getNow();
	}else{
		duration = getDuration(time);
	}

	qDebug("got frame %d, %d; time %f; packet size %d", got, cnt++, duration, pkt->size);

	if(got > 0){
        getImage(frame, image);
        av_frame_unref(frame);

		double duration = getDuration(starttime);

		QString name = m_cdcctx->codec->name;
		QString out = "decode (" + name + "):";

		m_durations[out] = duration;
	}

    av_frame_free(&frame);
}

#pragma warning(pop)

void RTSPServer::analyze_frame(AVFrame *frame, PImage& image)
{
//    if(m_frames.size() > m_max_frames)
//        return;

    if(frame->format == AV_PIX_FMT_YUV420P ||
            frame->format == AV_PIX_FMT_YUVJ420P ||
            frame->format == AV_PIX_FMT_NV12){

//      PImage obj;
		getImage(frame, image);
		m_image_updated = true;

//      m_mutex.lock();
//      m_frames.push(obj);
//      m_mutex.unlock();
    }
}

void RTSPServer::getImage(AVFrame *frame, PImage &obj)
{
    if(!obj.get())
        obj.reset(new Image);
    if(frame->format == AV_PIX_FMT_NV12){
        obj->setNV12(frame->data, frame->linesize, frame->width, frame->height);
    }else{
        obj->setYUV(frame->data, frame->linesize, frame->width, frame->height);
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

//bool RTSPServer::assemblyImages(AVPacket *pkt)
//{
//    if(pkt->size < rtp_packet_add_header::sizeof_header + 2)
//        return false;

//    size_t xOff, yOff, cntX, cntY;
//    unsigned short width, height;
//    if(!rtp_packet_add_header::getHeader(pkt->data + pkt->size - rtp_packet_add_header::sizeof_header - 2,
//                                     xOff, yOff, cntX, cntY, width, height)){
//        return false;
//    }
//    m_cntX = cntX;
//    m_cntY = cntY;
//    m_width = width;
//    m_height = height;
//    m_partImages.resize(m_cntX * m_cntY);
//    m_encodedData.resize(m_cntX * m_cntY);
//    size_t off = yOff * cntX + xOff;

//    if(!m_partImages[off].get()){
//        m_partImages[off].reset(new Image);
//    }
//    getEncodedData(pkt, m_encodedData[off]);
//    //decode_packet(pkt, m_partImages[off]);

//    //qDebug("%d %d %d %d %d", xOff, yOff, cntX, cntY, off);

//    if(off == m_partImages.size() - 1 && m_width && m_height){
//        assemplyOutput();
//    }
//    return true;
//}

//void RTSPServer::assemplyOutput()
//{
//    PImage obj;
//    obj.reset(new Image);
//    obj->setRGB((int)m_width, (int)m_height);

//    std::vector< pthread > threads;
//    threads.resize(m_encodedData.size());

////   #pragma omp parallel for num_threads(8)
//    for(size_t y = 0; y < m_encodedData.size(); ++y){
////        QFile f(QString::number(y) + ".jpeg");
////        if(f.open(QIODevice::WriteOnly)){
////            f.write((char*)m_encodedData[y].data(), m_encodedData[y].size());
////            f.close();
////        }
//        threads[y].reset(new std::thread([&](size_t t){
//            jpegenc dec;
//            dec.decode(m_encodedData[t], m_partImages[t]);
//        }, y));
//    }

//    for(int y = 0; y < m_encodedData.size(); ++y){
//        threads[y]->join();
//    }

//    std::vector<QPoint> offsets;
//    offsets.resize(m_encodedData.size());
//    int xOff = 0, yOff = 0, off = 0;
//    for(int y = 0; y < m_cntY; ++y){
//        xOff = 0;
//        for(int x = 0; x < m_cntX; ++x){
//            off = y * (int)m_cntX + x;

//            offsets[off] = QPoint(xOff, yOff);

//            xOff += (int)m_partImages[off]->width;
//        }
//        yOff += m_partImages[y * m_cntX]->height;
//    }

////#pragma omp parallel for num_threads(8)
//    for(int k = 0; k < m_cntX * m_cntY; ++k){
//        size_t x = k % m_cntX;
//        size_t y = k / m_cntX;
//        size_t off = y * m_cntX + x;
//        if(offsets[off].x() < obj->width && offsets[off].y() < obj->height)
//			copyRect(m_partImages[off], offsets[off].x(), offsets[off].y(), m_fvImage);
//    }
//	m_image_updated = true;

//	if(mRenderer){
//		mRenderer->loadImage(m_fvImage);
//	}
////    m_mutex.lock();
////    m_frames.push(obj);
////    m_mutex.unlock();

//    m_updatedImages = true;
//}

/**
 * @brief copyRect
 * copy image as part of other image
 * @param part
 * @param xOff
 * @param yOff
 * @param out
 */
void copyRect(const PImage &part, size_t xOff, size_t yOff, PImage &out)
{
    if(!part.get() || !out.get())
        return;
//    if(xOff + part->width > out->width || yOff + part->height > out->height)
//        return;

    if(part->type == Image::YUV){
        int linesizeYP = part->width;
        int linesizeUVP = part->width/2;
        int linesizeYO = out->width;
        int linesizeUVO = out->width/2;
		uchar *pY = part->yuv.data();
		uchar *oY = out->yuv.data();
		for(int y = 0; y < part->height; ++y){
			unsigned char * yp = pY + linesizeYP * y;
			unsigned char * yo = oY + linesizeYO * (y + yOff) + xOff;
            std::copy(yp, yp + linesizeYP, yo);
        }

		uchar *pU = pY + part->width * part->height;
		uchar *oU = oY + out->width * out->height;
		uchar *pV = pU + part->width/2 * part->height/2;
		uchar *oV = oU + out->width/2 * out->height/2;
		for(int y = 0; y < part->height/2; ++y){
			unsigned char * up = pU + linesizeUVP * y;
			unsigned char * uo = oU + linesizeUVO * (y + yOff/2) + xOff/2;

            std::copy(up, up + linesizeUVP, uo);

			unsigned char * vp = pV + linesizeUVP * y;
			unsigned char * vo = oV + linesizeUVO * (y + yOff/2) + xOff/2;
            std::copy(vp, vp + linesizeUVP, vo);
        }
    }else if(part->type == Image::RGB){
        int linesizeRGBP = part->width * 3;
        int linesizeRGBO = out->width * 3;

        int h = min(int(part->height), int(out->height - yOff));
        int w = min(int(part->width), int(out->width - xOff));
        int cp = w * 3;
        for(int y = 0; y < h; ++y){
            unsigned char * yp = part->rgb.data() + linesizeRGBP * y;
            unsigned char * yo = out->rgb.data() + linesizeRGBO * (y + yOff) + xOff * 3;
            std::copy(yp, yp + cp, yo);
        }
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
    if(!extratAddress(m_url, addr, port)){
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
    QStringList sl = QString(sdp).split("\n");

    for(QString s: sl){
        int pos = s.indexOf("m=video");
        if(pos >= 0){
            pos = s.indexOf("RTP/AVP");
            if(pos >= 0){
                QString fmt = s.right(s.size() - pos - QString("RTP/AVP").size()).trimmed();
                if(fmt == "26"){
                    m_idCodec = CODEC_JPEG;         /// select jpeg codec
                }else if(fmt == "96"){
                    m_idCodec = CODEC_H264;         /// select h264 codec
                }
            }
            m_state = SETUP;
            sendSetup();
            break;
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
    int ret = 0;

    if(m_idCodec == CODEC_H264){
		m_codec = avcodec_find_decoder_by_name(m_codecH264.toLatin1().data());
        if(!m_codec){
            m_codec = avcodec_find_decoder_by_name("h264");
            if(!m_codec){
                m_codec = avcodec_find_decoder_by_name("libx264");
                if(!m_codec){
                    m_error = "Codec not found";
                    return;
                }
            }
        }

        m_cdcctx = avcodec_alloc_context3(m_codec);
		m_cdcctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
		m_cdcctx->flags2 |= AV_CODEC_FLAG2_FAST;

		AVDictionary *dict = nullptr;
		av_dict_set(&dict, "threads", "auto", 0);
        av_dict_set(&dict, "zerolatency", "1", 0);

		if((ret = avcodec_open2(m_cdcctx, m_codec, &dict)) < 0){
            m_error = "Codec not open";
            avcodec_free_context(&m_cdcctx);
            return;
        }
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
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

#endif

    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr("0.0.0.0");
    addr.sin_port = htons(m_clientPort1);
    int res = bind(mHSocket, (SOCKADDR*)&addr, sizeof(addr));
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

					m_durations = mergeMaps(m_durations, m_ctpTransport.durations());

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
	if(!m_isStartDecode)
		return;
	//PImage obj;

    if(m_partImages.empty()){
        m_partImages.resize(1);
    }

	std::lock_guard<std::mutex> lg(m_mutexDecoder);

    double duration = -1;

	auto starttime = getNow();
	QString name;

    if(m_idCodec == CODEC_JPEG){
        if(m_useFastvideo){
			name = "Fastvideo";
			if(!m_decoderFv.get())
				m_decoderFv.reset(new fastvideo_decoder);

			m_decoderFv->decode((uchar*)enc.data(), enc.size(), m_fvImage, true);
			m_image_updated = true;
        }else{
			name = "JpegTurbo";
			jpegenc dec;
			dec.decode((uchar*)enc.data(), enc.size(), m_fvImage);
			m_image_updated = true;
		}
    }else{
		name = "";
        AVPacket pkt;
        av_init_packet(&pkt);
        av_new_packet(&pkt, enc.size());
        std::copy(enc.data(), enc.data() + enc.size(), pkt.data);
		decode_packet(&pkt, m_fvImage);
        av_packet_unref(&pkt);
		m_image_updated = true;
    }

	duration = getDuration(starttime);

	if(!name.isEmpty()){
		QString out = "decode (" + name + "):";
		m_durations[out] = duration;
	}

	updateRenderer();

    qDebug("decode duration(ms): %f         \r", duration);

//    if(!m_useFastvideo){
//        if(obj.get()){
//            m_mutex.lock();
//            m_frames.push(obj);
//            m_mutex.unlock();
//        }
//    }
}
