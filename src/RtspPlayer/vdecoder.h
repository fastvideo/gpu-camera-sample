#ifndef VDECODER_H
#define VDECODER_H

#include "common.h"

#include <mutex>

extern "C"{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

class CuvidDecoder;
class fastvideo_decoder;

class VDecoder
{
public:
    enum {
        NONE,
        CODEC_JPEG,
        CODEC_H264,
        CODEC_HEVC
    };

    VDecoder();
    ~VDecoder();

    bool initDecoder(bool use_stream = false);
    bool initContext(const QString &url, bool isClient = true);
    int initStream();
    int readPacket();
    void freePacket();

    QString error() const { return m_error; }
    /**
     * @brief sizeBufferUdp
     * @return
     */
    int sizeBufferUdp() const       { return m_bufferUdp; }
    /**
     * @brief setSizeBufferUdp
     * @param val
     */
    void setSizeBufferUdp(int val)  { m_bufferUdp = val; }
    /**
     * @brief setUseFastVideo
     * @param val
     */
    void setUseFastVideo(bool val)  {m_useFastvideo = val; }
    /**
     * @brief setH264Codec
     * @param codec
     */
    void setH264Codec(const QString& codec);
    void setCodec(int codec) { m_idCodec = codec; };

    bool isCuvidFound() const;
    bool isMJpeg() const;

    void setUseNvdecoder(bool val) {mUseNvDecoder = val;}

    bool decodePacket(PImage &image, QString &decodeName, quint64 &sizeReaded, double &duration);
    bool decodePacket(const QByteArray& enc, PImage& image, QString& decodeName, double &duration);
    void analyzeFrame(AVFrame *frame, PImage &image);
    void getEncodedData(AVPacket *pkt, bytearray& data);
    void getImage(AVFrame *frame, PImage &obj);
    /**
     * @brief decode_packet
     * @param pkt
     * @param customDecode
     */
    void waitUntilStopStreaming();
    void close();

    bool isDoStop() const;
    bool done() const;

private:
    AVCodec *m_codec = nullptr;
    AVFormatContext *m_fmtctx = nullptr;
    AVCodecContext *m_cdcctx = nullptr;
    AVInputFormat *m_inputfmt = nullptr;
    AVPacket m_pkt;
    bytearray mEncodedData;

    bool m_useFastvideo = false;
    QString m_codecH264 = "h264_cuvid";
    int m_idCodec = NONE;
    QString m_error;
    bool mUseNvDecoder = false;

    int m_bufferUdp = 5000000;

    std::mutex m_mutexDecoder;

    std::unique_ptr<fastvideo_decoder> m_decoderFv;
    std::unique_ptr<CuvidDecoder> mCuvidDecoder;

    bool m_done = false;
    bool m_is_open = false;
    bool m_doStop = false;
};

#endif // VDECODER_H
