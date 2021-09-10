#include "vdecoder.h"
#include <thread>
#include <QTime>
#include <mutex>

#include "fastvideo_decoder.h"
#include "jpegenc.h"

#ifndef __ARM_ARCH
#include "cuviddecoder.h"
#else
class CuvidDecoder{
public:
    enum ET{
        eH264,
        eHEVC
    };
    CuvidDecoder(ET et){}
    bool decode(uint8_t* data, size_t size, PImage& image){
        return false;
    }
};
#endif

const int STREAM_TYPE_VIDEO     = 0;
const int MAXIMUM_WAIT          = 1500;

int callback_layer(void* data)
{
    VDecoder *obj = static_cast<VDecoder*>(data);

    if(obj->done() || obj->isDoStop()){
        return 1;
    }

    return 0;
}

VDecoder::VDecoder()
{
    av_register_all();
    avformat_network_init();
    av_init_packet(&m_pkt);
}

VDecoder::~VDecoder()
{
    m_done = true;
    waitUntilStopStreaming();
    //av_packet_unref(&m_pkt);
    close();
}

bool VDecoder::initDecoder(bool use_stream)
{
    int ret;

    m_codec = nullptr;
    if(use_stream){
        if(m_idCodec == CODEC_JPEG){
            m_codec = avcodec_find_decoder_by_name("mjpeg");
        }
    }else{
        if(m_idCodec == CODEC_JPEG)
            return true;
        if(m_idCodec == CODEC_H264){
            m_codec = avcodec_find_decoder_by_name(m_codecH264.toLatin1().data());
            if(!m_codec){
                m_codec = avcodec_find_decoder_by_name("h264");
                if(!m_codec){
                    m_codec = avcodec_find_decoder_by_name("libx264");
                    if(!m_codec){
                        m_error = "Codec not found";
                        return false;
                    }
                }
            }
        }
        if(m_idCodec == CODEC_HEVC){
            m_codec = avcodec_find_decoder_by_name("hevc_cuvid");
            if(!m_codec){
                m_codec = avcodec_find_decoder_by_name("hevc");
                if(!m_codec){
                    m_codec = avcodec_find_decoder_by_name("libx265");
                    if(!m_codec){
                        m_error = "Codec not found";
                        return false;
                    }
                }
            }
        }
    }

    m_cdcctx = avcodec_alloc_context3(m_codec);
    if(use_stream){

        ret = avcodec_parameters_to_context(m_cdcctx, m_fmtctx->streams[STREAM_TYPE_VIDEO]->codecpar);
        if(ret < 0){
            m_error = QString("error copy paramters to context %1").arg(ret);
            m_is_open = false;
            return false;
        }
    }

    m_cdcctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
    m_cdcctx->flags2 |= AV_CODEC_FLAG2_FAST;

    if(use_stream){
        m_codec = avcodec_find_decoder(m_cdcctx->codec_id);
        if(!m_codec){
            m_error = "Codec not found";
            m_is_open = false;
            return false;
        }
        if(m_codec->id == AV_CODEC_ID_H264){
            AVCodec *c = avcodec_find_decoder_by_name(m_codecH264.toLatin1().data());
            if(c) m_codec = c;
            m_idCodec = CODEC_H264;
        }else if(m_codec->id == AV_CODEC_ID_HEVC){
            AVCodec *c = avcodec_find_decoder_by_name("hevc_cuvid");
            if(c) m_codec = c;
            m_idCodec = CODEC_HEVC;
        }else{
            m_idCodec = CODEC_JPEG;
        }
    }

    AVDictionary *dict = nullptr;
    av_dict_set(&dict, "threads", "auto", 0);
    av_dict_set(&dict, "zerolatency", "1", 0);
    av_dict_set_int(&dict, "buffer_size", m_bufferUdp, 0);

    if((ret = avcodec_open2(m_cdcctx, m_codec, &dict)) < 0){
        m_error = "Codec not open";
        avcodec_free_context(&m_cdcctx);
        return false;
    }

    return true;
}

bool VDecoder::initContext(const QString &url, bool isClient)
{
    int res;
    m_fmtctx = avformat_alloc_context();

    AVDictionary *dict = nullptr;

    if(!isClient){
        av_dict_set(&dict, "rtsp_flags", "listen", 0);
        qDebug("Try to open server %s ..", url.toLatin1().data());
    }else{
        qDebug("Try to open address %s ..", url.toLatin1().data());
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

        res = avformat_open_input(&m_fmtctx, url.toLatin1().data(), m_inputfmt, &dict);

        if(res < 0){
            qDebug("url %s can'not open", url.toLatin1().data());
            m_error = "url can'not open";
            m_is_open = false;
        }
    } catch (...) {
        m_is_open = false;
        m_error = "closed uncorrectrly";
        qDebug("closed input uncorrectly");
        return false;
    }
    return m_is_open;
}

int VDecoder::initStream()
{
    int res = 0;
    res = av_find_best_stream(m_fmtctx, AVMEDIA_TYPE_VIDEO, STREAM_TYPE_VIDEO, -1, nullptr, 0);

    if(res >= 0){
        res = avformat_find_stream_info(m_fmtctx, nullptr);

        res = initDecoder(true)? 0 : -1;
//        m_cdcctx = avcodec_alloc_context3(m_codec);
//        res = avcodec_parameters_to_context(m_cdcctx, m_fmtctx->streams[STREAM_TYPE_VIDEO]->codecpar);
//        if(res < 0){
//            m_error = QString("error copy paramters to context %1").arg(res);
//            m_is_open = false;
//            return res;
//        }
//        m_cdcctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
//        m_cdcctx->flags2 |= AV_CODEC_FLAG2_FAST;

//        m_codec = avcodec_find_decoder(m_cdcctx->codec_id);
//        if(!m_codec){
//            m_error = "Codec not found";
//            m_is_open = false;
//            return res;
//        }
//        if(m_codec->id == AV_CODEC_ID_H264){
//            AVCodec *c = avcodec_find_decoder_by_name(m_codecH264.toLatin1().data());
//            if(c) m_codec = c;
//            m_idCodec = CODEC_H264;
//        }

//        qDebug("Decoder found: %s", m_codec->name);

//        AVDictionary *dict = nullptr;
//        av_dict_set(&dict, "zerolatency", "1", 0);
//        av_dict_set_int(&dict, "buffer_size", m_bufferUdp, 0);

//        res = avcodec_open2(m_cdcctx, m_codec, &dict);
//        if(res < 0){
//            m_error = QString("decoder not open: %1").arg(res);
//            m_is_open = false;
//            return res;
//        }
    }
    return res;
}

int VDecoder::readPacket()
{
    int res;
    av_init_packet(&m_pkt);
    res = av_read_frame(m_fmtctx, &m_pkt);
    return res;
}

void VDecoder::freePacket()
{
    av_packet_unref(&m_pkt);
}

void VDecoder::setH264Codec(const QString &codec)
{
    m_codecH264 = codec;
    if(codec == "nv"){
        setUseNvdecoder(true);
    }
}

bool VDecoder::isCuvidFound() const
{
    return m_codec && (QString(m_codec->name) == "h264_cuvid" || QString(m_codec->name) == "hevc_cuvid");
}

bool VDecoder::isMJpeg() const
{
    return m_idCodec == CODEC_JPEG;
}

#pragma warning(push)
#pragma warning(disable : 4189)

bool VDecoder::decodePacket(PImage& image, QString& decodeName, quint64 &sizeReaded, double& duration)
{
    auto starttime = getNow();

    if(m_idCodec != CODEC_JPEG){
        int ret = 0;
        int got = 0;

        if(!mUseNvDecoder){
            if(!m_cdcctx)
                return false;
            AVFrame *frame = av_frame_alloc();

            ret = avcodec_decode_video2(m_cdcctx, frame, &got, &m_pkt);

            if(got > 0){
                analyzeFrame(frame, image);
                av_frame_unref(frame);

                decodeName = m_cdcctx->codec->name;
            }

            av_frame_free(&frame);
        }else{
            if(!mCuvidDecoder.get()){
                mCuvidDecoder.reset(new CuvidDecoder(m_idCodec == CODEC_H264? CuvidDecoder::eH264 : CuvidDecoder::eHEVC));
            }

            decodeName = "cuvid_nvcodec";
            mCuvidDecoder->decode(m_pkt.data, m_pkt.size, image);
        }
    }else{
//        PImage obj;

        getEncodedData(&m_pkt, mEncodedData);

        std::lock_guard<std::mutex> lg(m_mutexDecoder);
        if(m_useFastvideo){
            decodeName = "Fastvideo";
            if(!m_decoderFv.get())
                m_decoderFv.reset(new fastvideo_decoder);
            m_decoderFv->decode(mEncodedData, image, true);
        }else{
            decodeName = "JpegTurbo";
            jpegenc dec;
            dec.decode(mEncodedData, image);
        }


//        if(obj.get()){
//            m_mutex.lock();
//			m_frames.push(obj);
//            m_mutex.unlock();
//        }
    }

    if(!decodeName.isEmpty()){
        decodeName = "decode (" + decodeName + "):";
        duration = getDuration(starttime);
    }
    sizeReaded += m_pkt.size;
    return true;
}

bool VDecoder::decodePacket(const QByteArray &enc, PImage &image, QString &decodeName, double& duration)
{
    if(m_idCodec == CODEC_JPEG){
        auto starttime = getNow();
        if(m_useFastvideo){
            decodeName = "Fastvideo";
            if(!m_decoderFv.get())
                m_decoderFv.reset(new fastvideo_decoder);

            m_decoderFv->decode((uchar*)enc.data(), enc.size(), image, true);
        }else{
            decodeName = "JpegTurbo";
            jpegenc dec;
            dec.decode((uchar*)enc.data(), enc.size(), image);
        }

        if(!decodeName.isEmpty()){
            decodeName = "decode (" + decodeName + "):";
            duration = getDuration(starttime);
        }

    }else{
        quint64 tmp = 0;
        decodeName = "";
        av_init_packet(&m_pkt);
        av_new_packet(&m_pkt, enc.size());
        std::copy(enc.data(), enc.data() + enc.size(), m_pkt.data);
        decodePacket(image, decodeName, tmp, duration);
        av_packet_unref(&m_pkt);
    }
    return true;
}

#pragma warning(pop)

void VDecoder::analyzeFrame(AVFrame *frame, PImage &image)
{
    //    if(m_frames.size() > m_max_frames)
    //        return;

        if(frame->format == AV_PIX_FMT_YUV420P ||
                frame->format == AV_PIX_FMT_YUVJ420P ||
                frame->format == AV_PIX_FMT_NV12 ||
                frame->format == AV_PIX_FMT_P010){

            getImage(frame, image);
        }
}

void VDecoder::getEncodedData(AVPacket *pkt, bytearray &data)
{
    data.resize(pkt->size);
    std::copy(pkt->data, pkt->data + pkt->size, data.data());
}

void VDecoder::getImage(AVFrame *frame, PImage &obj)
{
    if(!obj.get())
        obj.reset(new RTSPImage);
    if(frame->format == AV_PIX_FMT_NV12){
        obj->setNV12(frame->data, frame->linesize, frame->width, frame->height);
    }else if(frame->format == AV_PIX_FMT_P010){
        obj->setP010(frame->data, frame->linesize, frame->width, frame->height);
    }else{
        obj->setYUV(frame->data, frame->linesize, frame->width, frame->height);
    }
}

void VDecoder::waitUntilStopStreaming()
{
    if(m_is_open){
        QTime time;
        time.start();
        m_doStop = true;
        while(m_is_open && time.elapsed() < MAXIMUM_WAIT){
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if(time.elapsed() > MAXIMUM_WAIT){
            qDebug("Oops. Streaming not stopped");
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

void VDecoder::close()
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
    m_is_open = false;
}

bool VDecoder::isDoStop() const
{
    return m_doStop;
}

bool VDecoder::done() const
{
    return m_done;
}
