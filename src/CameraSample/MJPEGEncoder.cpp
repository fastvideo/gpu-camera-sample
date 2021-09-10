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

#include "MJPEGEncoder.h"

extern "C"
{
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>
#include <libavformat/avformat.h>
}

MJPEGEncoder::MJPEGEncoder(int width,
                           int height,
                           int fps,
                           fastJpegFormat_t fmt,
                           const QString& outFileName)
{
    AVStream*       out_stream;
    AVCodecContext* enc_ctx;
    AVCodec*        encoder;
    unsigned int    i = 0;

    av_register_all();

    mFmtCtx = nullptr;
    mErr = avformat_alloc_output_context2(&mFmtCtx, nullptr, nullptr, outFileName.toStdString().c_str());
    if(!mFmtCtx)
    {
        qDebug("Could not create output context");
        return;
    }

    mFramesProcessed = 0;

    out_stream = avformat_new_stream(mFmtCtx, nullptr);
    if(!out_stream)
    {
        qDebug("Failed allocating output stream");
        mErr = AVERROR_UNKNOWN;
        return;
    }

    out_stream->time_base.den = fps;
    out_stream->time_base.num = 1;

    enc_ctx = out_stream->codec;
    encoder = avcodec_find_encoder(AV_CODEC_ID_MJPEG);

    enc_ctx->height = height;
    enc_ctx->width = width;
    enc_ctx->sample_aspect_ratio.den = 1;
    enc_ctx->sample_aspect_ratio.num = 1;

    enc_ctx->time_base.den = fps;
    enc_ctx->time_base.num = 1;

    enc_ctx->gop_size = 0; // all frames are intra frames

    switch (fmt)
    {
    case FAST_JPEG_Y:
        enc_ctx->pix_fmt = AV_PIX_FMT_GRAY8;
    case FAST_JPEG_444:
        enc_ctx->pix_fmt = AV_PIX_FMT_YUVJ444P;
        break;
    case FAST_JPEG_422:
        enc_ctx->pix_fmt = AV_PIX_FMT_YUVJ422P;
        break;
    case FAST_JPEG_420:
        enc_ctx->pix_fmt = AV_PIX_FMT_YUVJ420P;
        break;
    default:
        enc_ctx->pix_fmt = AV_PIX_FMT_YUVJ420P;
        break;
    }

    if(mFmtCtx->oformat->flags & AVFMT_GLOBALHEADER)
    {
        enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    /* Third parameter can be used to pass settings to encoder */
    mErr = avcodec_open2(enc_ctx, encoder, nullptr);
    if(mErr < 0)
    {
        qDebug("Cannot open video encoder for stream #%u", i);
        return;
    }

    if(mFmtCtx->oformat->flags & AVFMT_GLOBALHEADER)
        enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;


    if(!(mFmtCtx->oformat->flags & AVFMT_NOFILE))
    {
        mErr = avio_open(&mFmtCtx->pb, outFileName.toStdString().c_str(), AVIO_FLAG_WRITE);
        if(mErr < 0)
        {
            qDebug("Could not open output file '%s'", outFileName.toLocal8Bit().data());
            return;
        }
    }

    /* init muxer, write output file header */
    mErr = avformat_write_header(mFmtCtx, nullptr);
    if(mErr < 0)
    {
        qDebug("Error occurred when opening output file");
        return;
    }

}

MJPEGEncoder::~MJPEGEncoder()
{
    close();
}

bool MJPEGEncoder::addJPEGFrame(unsigned char *jpgPtr, int jpgSize)
{
    if(mErr < 0)
        return false;

    QMutexLocker l(&mLock);

    bool ret = false;
    AVCodecContext* pVideoCodec = mFmtCtx->streams[0]->codec;
    if(pVideoCodec->codec_id != AV_CODEC_ID_MJPEG )
        return ret;

    int err;
    char errbuf[1024];

    //Write video frame
    AVPacket videoPkt;
    av_init_packet(&videoPkt);

    videoPkt.flags |= AV_PKT_FLAG_KEY;
    videoPkt.stream_index = 0; //Output video stream
    videoPkt.data= jpgPtr;
    videoPkt.size = jpgSize;
    videoPkt.pts = mFramesProcessed;
    videoPkt.dts = mFramesProcessed;

    mFramesProcessed++;

    err = av_interleaved_write_frame(mFmtCtx, &videoPkt);
    if(err != 0)
    {
        av_strerror(err, errbuf, sizeof(errbuf));
        qDebug("Error writing video frame: %s", errbuf);
    }
    else
    {
        pVideoCodec->frame_number++;
        ret = true;
    }
    av_free_packet(&videoPkt);


    return ret;
}

void MJPEGEncoder::close()
{
    if(mErr < 0)
        return;

    QMutexLocker l(&mLock);

    av_write_trailer(mFmtCtx);
    if(mFmtCtx && mFmtCtx->nb_streams > 0)
    {
        for(unsigned i = 0; i < mFmtCtx->nb_streams; i++)
        {
            if(mFmtCtx->streams[i] && mFmtCtx->streams[i]->codec)
                avcodec_close(mFmtCtx->streams[i]->codec);
        }
    }


    if(mFmtCtx && !(mFmtCtx->oformat->flags & AVFMT_NOFILE))
        avio_close(mFmtCtx->pb);
    avformat_free_context(mFmtCtx);

    mErr = -1;
}
