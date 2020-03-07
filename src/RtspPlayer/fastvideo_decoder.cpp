#include "fastvideo_decoder.h"

#include <QFile>

#include <helper_jpeg/helper_jpeg.hpp>

fastvideo_decoder::fastvideo_decoder()
{
    fastStatus_t ret;
    ret = fastInit(1, false);

    //fastSdkParametersHandle_t params;
}

fastvideo_decoder::~fastvideo_decoder()
{
    if(m_handle)
        fastJpegDecoderDestroy(m_handle);
    if(m_DeviceToHost)
        fastExportToHostDestroy(m_DeviceToHost);
}

bool fastvideo_decoder::init_decoder(uint32_t width, uint32_t height, fastSurfaceFormat_t fmt)
{
    Q_UNUSED(height)
    fastStatus_t ret;

	ret = fastJpegDecoderCreate(&m_handle, fmt, width, width, true, &m_dHandle);

    ret = fastExportToHostCreate(&m_DeviceToHost, &m_surfaceFmt, m_dHandle);

    unsigned reqMem = 0;
    ret = fastJpegDecoderGetAllocatedGpuMemorySize(m_handle, &reqMem);
    qDebug("requested memory %d", reqMem);

    return ret == FAST_OK;
}

bool fastvideo_decoder::decode(const uint8_t *input, uint32_t len, PImage &output)
{
    fastJfifInfo_t info;
    fastStatus_t ret;
    fastExportParameters_t params;

    buf.resize(len);
    unsigned char *h_Bytestream = buf.data();

    info.bytestreamSize = (unsigned)buf.size();
    info.h_Bytestream = h_Bytestream;

//    QFile f("out.jpg");
//    f.open(QIODevice::WriteOnly);
//    f.write((char*)input.data(), input.size());
//    f.close();

    ret = fastJfifLoadFromMemory(input, len, &info);

	fastSurfaceFormat_t fmt = info.jpegFmt == JPEG_Y? FAST_I8 : FAST_RGB8;

    if(!m_isInit){
		m_isInit = init_decoder(info.width, info.height, fmt);
    }

    if(ret != FAST_OK || !m_isInit)
        return false;

    ret = fastJpegDecode(m_handle, &info);

    if(ret == FAST_OK){
        if(!output.get() || output->width != info.width || output->height != info.height){
			output.reset(new Image(info.width, info.height, fmt == FAST_I8? Image::GRAY : Image::RGB));
        }

		int channels = fmt == FAST_I8? 1 : 3;

		int pitch = output->width * channels;

        ret = fastExportToHostCopy(m_DeviceToHost, output->rgb.data(), output->width, pitch,
                                   output->height, &params);
        return ret == FAST_OK;
    }
    return false;
}

bool fastvideo_decoder::decode(const bytearray &input, PImage& output)
{
    if(!m_handle || !m_DeviceToHost)
        return false;

    fastJfifInfo_t info;
    fastStatus_t ret;
    fastExportParameters_t params;

    buf.resize(input.size());
    unsigned char *h_Bytestream = buf.data();

    info.bytestreamSize = (unsigned)buf.size();
    info.h_Bytestream = h_Bytestream;

//    QFile f("out.jpg");
//    f.open(QIODevice::WriteOnly);
//    f.write((char*)input.data(), input.size());
//    f.close();

    ret = fastJfifLoadFromMemory(input.data(), (unsigned)input.size(), &info);

    if(ret != FAST_OK)
        return false;

    ret = fastJpegDecode(m_handle, &info);

    if(ret == FAST_OK){
        if(!output.get() || output->width != info.width || output->height != info.height){
            qDebug("new alloc");
            output.reset(new Image(info.width, info.height, Image::RGB));
        }

        int pitch = output->width * 3;

        ret = fastExportToHostCopy(m_DeviceToHost, output->rgb.data(), output->width, pitch,
                                   output->height, &params);
        return ret == FAST_OK;
    }
    return false;
}
