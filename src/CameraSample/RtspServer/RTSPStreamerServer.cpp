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

#include "RTSPStreamerServer.h"

#include <QRect>

#include <thread>
#include <exception>
#include <cmath>

#include "common_utils.h"
#include "vutils.h"

#include <QPainter>
#include <QImage>
#include <QDateTime>

RTSPStreamerServer::RTSPStreamerServer(int width, int height,
                                       int channels,
                                       const QString &url,
                                       EncoderType encType,
                                       unsigned bitrate,
                                       QObject *parent)
	: QObject(parent)
    , mWidth(width)
    , mHeight(height)
    , mChannels(channels)
    , mEncoderType(encType)
    , mBitrate(bitrate)
    , mUrl(url)
{
	avcodec_register_all();
	av_register_all();
    avformat_network_init();

    mJpegEncode = encodeJpeg;

    if(mEncoderType == etNVENC)
    {
#ifdef __ARM_ARCH
        mV4L2Encoder.reset(new v4l2Encoder());
        mV4L2Encoder->setIDRInterval(1);
        mV4L2Encoder->setEnableAllIFrameEncode(true);
        mV4L2Encoder->setInsertSpsPpsAtIdrEnabled(true);
        mV4L2Encoder->setInsertVuiEnabled(true);
        mV4L2Encoder->setIFrameInterval(1);
//        mCodec = avcodec_find_encoder_by_name("h264_v4l2m2m");
#else
        mCodec = avcodec_find_encoder_by_name("h264_nvenc");
#endif
        if(!mCodec)
        {
            mCodec = avcodec_find_encoder_by_name("libx264");
            if(!mCodec)
            {
                mIsError = true;
                mErrStr = "Codec not found";
                return;
			}
        }
        mPixFmt = AV_PIX_FMT_NV12;
    }else if(mEncoderType == etNVENC_HEVC){
#ifdef __ARM_ARCH
        mV4L2Encoder.reset(new v4l2Encoder());
        mV4L2Encoder->setEncoder(v4l2Encoder::eHEVC);
        mV4L2Encoder->setIDRInterval(1);
        mV4L2Encoder->setEnableAllIFrameEncode(true);
        mV4L2Encoder->setInsertSpsPpsAtIdrEnabled(true);
        mV4L2Encoder->setInsertVuiEnabled(true);
        mV4L2Encoder->setIFrameInterval(1);
//        mCodec = avcodec_find_encoder_by_name("h264_v4l2m2m");
#else
        mCodec = avcodec_find_encoder_by_name("hevc_nvenc");
#endif
        if(!mCodec)
        {
            mCodec = avcodec_find_encoder_by_name("libx265");
            if(!mCodec)
            {
                mCodec = avcodec_find_encoder(AV_CODEC_ID_HEVC);
                if(!mCodec){
                    mIsError = true;
                    mErrStr = "Codec not found";
                return;
                }
            }
        }
        //mPixFmt = AV_PIX_FMT_NV12;

    }else
    if(mEncoderType == etJPEG)
    {
        mCodec = avcodec_find_encoder_by_name("mjpeg");
        if(!mCodec)
        {
            mIsError = true;
			mErrStr = "Codec not found";
			return;
		}

        mPixFmt = AV_PIX_FMT_YUVJ420P;
    }

    mCodecId = mCodec->id;

    mCtx = avcodec_alloc_context3(mCodec);

    mCtx->bit_rate = mBitrate;

    if(mEncoderType == etJPEG)
    {
        if(mWidth > MAX_WIDTH_RTP_JPEG || mHeight > MAX_HEIGHT_RTP_JPEG)
        {
            mCtx->width = MAX_WIDTH_JPEG;
            mCtx->height = MAX_HEIGHT_JPEG;
        }
        else
        {
            mCtx->width = mWidth;
            mCtx->height = mHeight;
        }
    }
    else
    {
        mCtx->width = mWidth;
        mCtx->height = mHeight;
	}

	//frames per second
    mCtx->time_base = {1, 60};         // for test. maybe do not affect
    mCtx->framerate = {60, 1};         // for test. maybe do not affect
	mCtx->gop_size = 0;
    mCtx->pix_fmt = mPixFmt;

    if(mEncoderType == etNVENC || mEncoderType == etNVENC_HEVC)
    {
        mCtx->max_b_frames = 0;        // codec do not open for mjpeg
		mCtx->keyint_min = 0;
        mCtx->flags |= AV_CODEC_FLAG_LOW_DELAY;
		mCtx->flags2 |= AV_CODEC_FLAG2_FAST;
    }

	AVDictionary *dict = nullptr;
	av_dict_set(&dict, "c", "v", 0);
    if(mCodecId == AV_CODEC_ID_MJPEG)
    {
        av_dict_set(&dict, "q:v", "3", 0);
		av_dict_set(&dict, "huffman", "0", 0);                      // need for mjpeg
		av_dict_set(&dict, "force_duplicated_matrix", "1", 0);      // remove warnings where mjpeg sending
	}

    if(mCodecId == AV_CODEC_ID_H264 || mCodecId == AV_CODEC_ID_HEVC || mCodecId == AV_CODEC_ID_INDEO3)
    {
        av_dict_set(&dict, "zerolatency", "1", 0);
        //av_dict_set(&dict, "preset", "fast", 0);
		av_dict_set(&dict, "movflags", "+faststart", 0);
        av_dict_set(&dict, "delay", "0", 0);
        av_dict_set(&dict, "rc", "cbr_ld_hq", 0);
	}

    int ret = avcodec_open2(mCtx, mCodec, &dict);
	if(ret < 0)
	{
		char buf[100];
		av_make_error_string(buf, sizeof(buf), ret);
        if(mEncoderType != etJPEG){
            mErrStr = QString(QStringLiteral("avcodec_open2 failed, code: %1 (%2)")).arg(ret, 0, 16).arg(buf);
            mIsError = true;
            return;
        }else{
            qDebug("Can use only ctp protocol");
        }
    }

	mDelayFps = 1000 / mFps;

	mTimerCtrlFps.restart();

    mEncoderBuffer.resize(mWidth * mHeight * 4);
}

RTSPStreamerServer::~RTSPStreamerServer()
{
	mDone = true;
	if(mFrameThread.get()){
		mFrameThread->join();
		mFrameThread.reset();
	}

    if(mThread.get())
    {
        mThread->quit();
        mThread->wait();
	}
    try{
        if(mCtx)
        {
            avcodec_close(mCtx);
            avcodec_free_context(&mCtx);
        }
    }catch(...){
        qDebug("unknown error when close codec and context");
    }
}

void RTSPStreamerServer::setBitrate(qint64 bitrate)
{
    mBitrate = bitrate;
}

void RTSPStreamerServer::setEncodeFun(TEncodeRgb fun)
{
    mJpegEncode = fun;
}

void RTSPStreamerServer::setEncodeNv12Fun(TEncodeNv12 fun)
{
    mNv12Encode = fun;
}

void RTSPStreamerServer::setMultithreading(bool val)
{
    mMultithreading = val;
}

bool RTSPStreamerServer::multithreading() const
{
    return mMultithreading;
}

void RTSPStreamerServer::setUseCustomEncodeJpeg(bool val)
{
    mUseCustomEncodeJpeg = val;
}

void RTSPStreamerServer::setUseCustomEncodeH264(bool val)
{
    mUseCustomEncodeH264 = val;
}

bool RTSPStreamerServer::isError() const
{
    return mIsError;
}

QString RTSPStreamerServer::errorStr() const
{
    return mErrStr;
}

bool RTSPStreamerServer::isConnected() const
{
	return mIsInitialized && !mClients.empty() && isAnyClientInit();
}

bool RTSPStreamerServer::isAnyClientInit() const
{
	for(TcpClient *c: mClients){
		if(c->isInit()){
			return true;
		}
	}
	return false;
}

bool RTSPStreamerServer::isStarted() const
{
    return  mServer.get() && mServer->isListening();
}

bool RTSPStreamerServer::startServer()
{
    if(mIsError)
		return false;

    if(!mWidth || !mHeight || !mChannels || mUrl.isEmpty())
    {
        qDebug("did not set parameters");
		return false;
	}
    QString addr, port, url = mUrl;
    if(url.indexOf("rtsp://") >= 0)
    {
		url = url.remove(0, 7);
		int pos = url.indexOf(":", 0);
        if(pos >= 0)
        {
			addr = url.left(pos);

			url = url.remove(0, pos + 1);
			pos = url.indexOf("/");
            if(pos >= 0)
            {
				port = url.left(pos);
            }
            else
            {
				port = url;
			}
            mHost = QHostAddress(addr);
            mPort = port.toUInt();

            if(!mPort)
            {
                qDebug("wrong url. port");
                mIsError = true;
				return false;
			}

            mThread.reset(new QThread);
            mThread->setObjectName("RTSP Server thread");
            moveToThread(mThread.get());
            mThread->start();

			QTimer::singleShot(0, this, [this](){
				doServer();
			});
			return true;
        }
        else
        {
            qDebug("wrong url. port");
            mIsError = true;
			return false;
		}
    }
    else
    {
        qDebug("wrong url. name");
        mIsError = true;
		return false;
	}
    mIsError = true;
    return false;
}

double RTSPStreamerServer::duration() const
{
    return mDuration;
}

void RTSPStreamerServer::removeClient(TcpClient *client)
{
    for(auto it = mClients.begin(); it != mClients.end(); ++it)
    {
        if(*it == client)
        {
            it = mClients.erase(it);
			client->deleteLater();
			break;
		}
	}
}

void RTSPStreamerServer::newConnection()
{
    QTcpSocket* sock = mServer->nextPendingConnection();
    if(sock)
    {
        TcpClient *client = new TcpClient(sock, mUrl, mCtx, (TcpClient::EncoderType)mEncoderType);
        mClients.push_back(client);
		connect(client, SIGNAL(removeClient(TcpClient*)), this, SLOT(removeClient(TcpClient*)));
//		connect(sock, SIGNAL(disconnected()),
//				client, SLOT(deleteLater()), Qt::QueuedConnection);
        qDebug("new connection: %s:%d", sock->peerAddress().toString().toLatin1().data(), sock->peerPort());
		mIsInitialized = true;
	}
}

void RTSPStreamerServer::doServer()
{
    mServer.reset(new QTcpServer);

    qDebug("---- server start -----");
    mServer->listen(mHost, mPort);
    connect(mServer.get(), SIGNAL(newConnection()), this, SLOT(newConnection()));
}

void RTSPStreamerServer::RGB2Yuv420p(unsigned char *yuv,
							   unsigned char *rgb,
							   int width,
							   int height)
{
  const size_t image_size = width * height;
  unsigned char *dst_y = yuv;
  unsigned char *dst_u = yuv + image_size;
  unsigned char *dst_v = yuv + image_size * 5 / 4;

	// Y plane
	for(size_t i = 0; i < image_size; i++)
	{
		int r = rgb[3 * i];
		int g = rgb[3 * i + 1];
		int b = rgb[3 * i + 2];
		*dst_y++ = ((67316 * r + 132154 * g + 25666 * b) >> 18 ) + 16;
	}

	// U and V plane
	for(size_t y = 0; y < height; y+=2)
	{
		for(size_t x = 0; x < width; x+=2)
		{
			const size_t i = y * width + x;
			int r = rgb[3 * i];
			int g = rgb[3 * i + 1];
			int b = rgb[3 * i + 2];
			*dst_u++ = ((-38856 * r - 76282 * g + 115138 * b ) >> 18 ) + 128;
			*dst_v++ = ((115138 * r - 96414 * g - 18724 * b) >> 18 ) + 128;
		}
	}
}

void RTSPStreamerServer::Gray2Yuv420p(unsigned char *yuv, unsigned char *gray, int width, int height)
{
	const size_t image_size = width * height;
	unsigned char *dst_y = yuv;
	unsigned char *dst_u = yuv + image_size;
	unsigned char *dst_v = yuv + image_size * 5 / 4;

	  // Y plane
	  for(size_t i = 0; i < image_size; i++)
	  {
		  int r = gray[i];
		  *dst_y++ = ((67316 * r + 132154 * r + 25666 * r) >> 18 ) + 16;
	  }

	  // U and V plane
	  for(size_t y = 0; y < height; y+=2)
	  {
		  for(size_t x = 0; x < width; x+=2)
		  {
			  const size_t i = y * width + x;
			  int r = gray[i];
			  *dst_u++ = ((-38856 * r - 76282 * r + 115138 * r ) >> 18 ) + 128;
			  *dst_v++ = ((115138 * r - 96414 * r - 18724 * r) >> 18 ) + 128;
		  }
	  }
}

bool RTSPStreamerServer::addBigFrame(unsigned char* rgbPtr, size_t linesize)
{
    if(!mIsInitialized || mClients.empty())
		return false;

    if(mEncoderType != etJPEG || (mWidth <= MAX_WIDTH_RTP_JPEG && mHeight <= MAX_HEIGHT_RTP_JPEG))
		return addFrame(rgbPtr);

    size_t cntW = std::round(1. * mWidth/MAX_WIDTH_JPEG), cntH = std::round(mHeight/MAX_HEIGHT_JPEG);
    while(cntW * MAX_WIDTH_JPEG < mWidth)cntW++;
    while(cntH * MAX_HEIGHT_JPEG < mHeight)cntH++;
	const size_t cntAll = cntW * cntH;

	if(!linesize)
        linesize = mWidth * mChannels;

    mData.resize(cntAll);

    size_t xOff = 0, yOff = 0, w = std::min((size_t)mWidth, MAX_WIDTH_JPEG), h = std::min((size_t)mHeight, MAX_HEIGHT_JPEG);
	std::vector< QRect > sizes;
	sizes.resize(cntAll);
    for(size_t y = 0; y < cntH; ++y)
    {
        h = std::min((size_t)mHeight - yOff, MAX_HEIGHT_JPEG);
		xOff = 0;
        for(size_t x = 0; x < cntW; ++x)
        {
            w = std::min((size_t)mWidth - xOff, MAX_WIDTH_JPEG);
			size_t k = y * cntW + x;
            sizes[k] = QRect(static_cast<int>(xOff), static_cast<int>(yOff),
                             static_cast<int>(w), static_cast<int>(h));
			xOff += w;
		}
		yOff += h;
	}

//#pragma omp parallel for num_threads(8)
    for(size_t k = 0; k < cntAll; ++k)
    {
        //size_t w = static_cast<size_t>(sizes[k].width());
        size_t h = static_cast<size_t>(sizes[k].height());

        mData[k].resize(MAX_WIDTH_JPEG * MAX_HEIGHT_JPEG * mChannels);
//        m_yuv[k].resize(w * h + w/2 * h/2 * 2);

        copyPartImage(rgbPtr, sizes[k].x(), sizes[k].y(), mChannels, linesize, h, MAX_WIDTH_JPEG * mChannels, mData[k].data());
//        RGB2Yuv420p(m_yuv[k].data(), m_data[k].data(), w, h);
	}

    std::vector<AVPacket> pkts;
	pkts.resize(cntAll);
    mJpegData.resize(cntAll);

    auto fun = [&](size_t t)
    {
        av_init_packet(&pkts[t]);
        pkts[t].pts = mFramesProcessed + t;

        mJpegEncode(static_cast<int>(t), mData[t].data(), MAX_WIDTH_JPEG, MAX_HEIGHT_JPEG, mChannels, mJpegData[t]);

		av_new_packet(&pkts[t], static_cast<int>(mJpegData[t].size + rtp_packet_add_header::sizeof_header));
        pkts[t].pts = mFramesProcessed + t;

        uchar x = static_cast<uchar>(t % cntW);
        uchar y = static_cast<uchar>(t / cntW);

		std::copy(mJpegData[t].buffer.data(), mJpegData[t].buffer.data() + mJpegData[t].size, pkts[t].data);
        rtp_packet_add_header::setHeader(pkts[t].data + pkts[t].size - rtp_packet_add_header::sizeof_header - 2,
                                         x, y, cntW, cntH, mWidth, mHeight);
    };

    if(mMultithreading)
    {
        std::vector<pthread> threads;
        threads.resize(mJpegData.size());

    //#pragma omp parallel for num_threads(4)
        for(size_t k = 0; k < cntAll; ++k)
        {
            threads[k].reset(new std::thread(fun, k));
        }

        for(int y = 0; y < cntAll; ++y)
        {
            threads[y]->join();
            threads[y].reset();
        }
    }
    else
    {
        for(size_t k = 0; k < cntAll; ++k)
        {
            fun(k);
        }
    }

    for(int k = 0; k < cntAll; ++k)
    {
        sendPkt(&pkts[k]);
		av_packet_unref(&pkts[k]);
	}

	mFramesProcessed += cntAll + 1;

	return true;
}

bool RTSPStreamerServer::addFrame(unsigned char *rgbPtr)
{
//	if(mTimerCtrlFps.elapsed() - mDelayFps < mCurrentTimeElapsed){
//		return false;
//	}
//	mCurrentTimeElapsed = mTimerCtrlFps.elapsed();
	// unsafe but push buffer to another thread
	std::lock_guard<std::mutex> lg(mFrameMutex);

	if(mFrameBuffers.size() < mMaxFrameBuffers)
		mFrameBuffers.push_back(FrameBuffer(rgbPtr));

	if(!mFrameThread.get()){
		mFrameThread.reset(new std::thread([this](){
			doFrameBuffer();
		}));
	}
	return true;
}

void RTSPStreamerServer::doFrameBuffer()
{
	while(!mDone){
		if(mFrameBuffers.empty()){
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}else{
			mFrameMutex.lock();
			FrameBuffer fb = mFrameBuffers.front();
			mFrameBuffers.pop_front();
			mFrameMutex.unlock();

			addInternalFrame(fb.buffer);
		}
	}
}

void drawTimeToImage(uchar *rgbPtr, int width, int height, const QDateTime& time)
{
    if(!rgbPtr)
        return;
    QImage image(rgbPtr, width, height, QImage::Format_RGB32);
    QPainter painter(&image);

    painter.setFont(QFont("Arial", 80));
    QPen pen;
    pen.setWidth(2);
    pen.setColor(Qt::white);
    painter.setPen(pen);
    painter.drawText(10, 100, time.toString("hh:mm:ss.zzz"));
    qDebug("time %s", time.toString("hh:mm:ss.zzz").toLatin1().data());
}

void drawTimeToImageGray(uchar *gray, int width, int height, const QDateTime& time)
{
    if(!gray)
        return;
    QImage image(gray, width, height, QImage::Format_Grayscale8);
    QPainter painter(&image);

    painter.setFont(QFont("Arial", 80));
    QPen pen;
    pen.setWidth(2);
    pen.setColor(Qt::white);
    painter.setPen(pen);
    painter.drawText(10, 100, time.toString("hh:mm:ss.zzz"));
    qDebug("time %s", time.toString("hh:mm:ss.zzz").toLatin1().data());
}

bool RTSPStreamerServer::addInternalFrame(uchar *rgbPtr)
{
	auto starttime = getNow();

    if(!mIsInitialized || mClients.empty())
        return false;
	int ret = 0;

    if(((mCodecId == AV_CODEC_ID_H264 || mCodecId == AV_CODEC_ID_INDEO3 || mCodecId == AV_CODEC_ID_HEVC) && !mUseCustomEncodeH264)
            || (mEncoderType == etJPEG && !mUseCustomEncodeJpeg))
	{
		AVFrame* frm = av_frame_alloc();
		frm->width = mWidth;
		frm->height = mHeight;
		frm->format = mPixFmt;
		frm->pts = mFramesProcessed++;
		//Set frame->data pointers manually
		if(mChannels == 1)
		{
			Gray2Yuv420p(mEncoderBuffer.data(), rgbPtr, mWidth, mHeight);
		}
		else
		{
//            QDateTime dt = QDateTime::currentDateTime();
//            drawTimeToImage(rgbPtr, mWidth, mHeight, dt);

            if(mNv12Encode != nullptr && mPixFmt == AV_PIX_FMT_NV12){
                frm->format = AV_PIX_FMT_NV12;
                mNv12Encode(mEncoderBuffer.data(), rgbPtr, mWidth, mHeight);
//                drawTimeToImageGray(mEncoderBuffer.data(), mWidth, mHeight, dt);
            }else{
                RGB2Yuv420p(mEncoderBuffer.data(), rgbPtr, mWidth, mHeight);
            }
        }
#ifdef __ARM_ARCH
        if(mEncoderType == etNVENC || mEncoderType == etNVENC_HEVC){
            encodeWriteFrame(mEncoderBuffer.data(), mWidth, mHeight);
        }else
#endif
        {
            ret = av_image_fill_arrays(frm->data, frm->linesize, mEncoderBuffer.data(), (AVPixelFormat)frm->format, frm->width, frm->height, 1);
//      	ret = encode_write_frame(frm, 0, &got_frame);
            encodeWriteFrame(frm);
        }

		av_frame_free(&frm);
	}
	else
	{
		AVPacket pkt;
		av_init_packet(&pkt);

		int t = 0;

        if(mEncoderType == etJPEG)
		{
			if(mJpegData.empty())
				mJpegData.resize(1);
			mJpegEncode(t, rgbPtr, mWidth, mHeight, mChannels, mJpegData[t]);
		}
		else
		{
			throw new std::exception();
		}

		av_new_packet(&pkt, static_cast<int>(mJpegData[t].size));
		pkt.pts = mFramesProcessed++;

		std::copy(mJpegData[t].buffer.data(), mJpegData[t].buffer.data() + mJpegData[t].size, pkt.data);

		sendPkt(&pkt);
		av_packet_unref(&pkt);
	}

	double duration = getDuration(starttime);
	qDebug("encode duration %f", duration);
    mDuration = duration;

	if(ret == 0)
	{
		mFramesProcessed++;
		return true;
	}

	return false;
}

#ifdef __ARM_ARCH
void RTSPStreamerServer::encodeWriteFrame(uint8_t *buf, int width, int height)
{
    if(mV4L2Encoder.data()){
        if(mV4L2Encoder->encodeFrame(buf, width, height, mUserBuffer, mEncoderType == etNVENC_HEVC)){
            if(!mUserBuffer.empty()){
                AVPacket enc_pkt;
                enc_pkt.data = nullptr;
                enc_pkt.size = 0;
                av_init_packet(&enc_pkt);

                av_new_packet(&enc_pkt, static_cast<int>(mUserBuffer.size()));
                enc_pkt.pts = enc_pkt.dts = mFramesProcessed;
                enc_pkt.flags = AV_PKT_FLAG_KEY;
                std::copy(mUserBuffer.data(),mUserBuffer.data() + mUserBuffer.size(), enc_pkt.data);

                sendPkt(&enc_pkt);
                av_packet_unref(&enc_pkt);
            }
        }
    }
}
#endif

void RTSPStreamerServer::encodeWriteFrame(AVFrame *frame)
{
    int ret = 0, got;
	AVPacket enc_pkt;
    enc_pkt.data = nullptr;
	enc_pkt.size = 0;

    av_init_packet(&enc_pkt);

    ret = avcodec_encode_video2(mCtx, &enc_pkt, frame, &got);
    if(got > 0){
        enc_pkt.pts = frame->pts;
        sendPkt(&enc_pkt);
        av_packet_unref(&enc_pkt);
    }
}

void RTSPStreamerServer::sendPkt(AVPacket *pkt)
{
    for(TcpClient *c: mClients){
		c->sendpkt(pkt);
	}
}
