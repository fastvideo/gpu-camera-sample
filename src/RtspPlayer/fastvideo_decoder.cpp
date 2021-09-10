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
	release_decoder();
}

bool fastvideo_decoder::init_decoder(uint32_t width, uint32_t height, fastSurfaceFormat_t fmt, bool cudaImage)
{
    Q_UNUSED(height)
    fastStatus_t ret;

	release_decoder();

	ret = fastJpegDecoderCreate(&m_handle, fmt, width, width, true, &m_dHandle);

	if(cudaImage){
		ret = fastExportToDeviceCreate(&m_DeviceToDevice, &m_surfaceFmt, m_dHandle);
	}else{
		ret = fastExportToHostCreate(&m_DeviceToHost, &m_surfaceFmt, m_dHandle);
	}

        size_t allMem = 0;
	{
                size_t reqMem = 0;
		ret = fastJpegDecoderGetAllocatedGpuMemorySize(m_handle, &reqMem);
		allMem += reqMem;
	}

	{
                size_t reqMem = 0;
		ret = fastExportToDeviceGetAllocatedGpuMemorySize(m_DeviceToDevice, &reqMem);
		allMem += reqMem;
	}

	qDebug("requested memory %d", allMem);

	m_width = width;
	m_height = height;

	return ret == FAST_OK;
}

void fastvideo_decoder::release_decoder()
{
	if(m_handle)
		fastJpegDecoderDestroy(m_handle);
	if(m_DeviceToHost)
		fastExportToHostDestroy(m_DeviceToHost);
	if(m_DeviceToDevice)
		fastExportToDeviceDestroy(m_DeviceToDevice);
	m_handle = nullptr;
	m_DeviceToHost = nullptr;
	m_DeviceToDevice = nullptr;
}

bool fastvideo_decoder::decode(const uint8_t *input, uint32_t len, PImage &output, bool cudaImage)
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

        fastSurfaceFormat_t fmt = info.jpegFmt == FAST_JPEG_Y? FAST_I8 : FAST_RGB8;

	m_isInit &= info.width == m_width && info.height == m_height;

	bool reinit = false;
	if(!m_isInit || m_useCuda != cudaImage){
		m_isInit = init_decoder(info.width, info.height, fmt, cudaImage);
		m_useCuda = cudaImage;
		reinit = true;
    }

    if(ret != FAST_OK || !m_isInit)
        return false;

    ret = fastJpegDecode(m_handle, &info);

    if(ret == FAST_OK){
		if(cudaImage){
			if(reinit || !output.get() || output->width != info.width || output->height != info.height){
                                output.reset(new RTSPImage(info.width, info.height, fmt == FAST_I8? RTSPImage::CUDA_GRAY : RTSPImage::CUDA_RGB));
			}
		}else{
			if(reinit || !output.get() || output->width != info.width || output->height != info.height){
                                output.reset(new RTSPImage(info.width, info.height, fmt == FAST_I8? RTSPImage::GRAY : RTSPImage::RGB));
			}
		}

		int channels = fmt == FAST_I8? 1 : 3;

		int pitch = output->width * channels;

		if(cudaImage){
			ret = fastExportToDeviceCopy(m_DeviceToDevice, output->cudaRgb, output->width, pitch,
										 output->height, &params);
		}else{
			ret = fastExportToHostCopy(m_DeviceToHost, output->rgb.data(), output->width, pitch,
									   output->height, &params);
		}
        return ret == FAST_OK;
    }
    return false;
}

bool fastvideo_decoder::decode(const bytearray &input, PImage& output, bool cudaImage)
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

	m_isInit &= info.width == m_width && info.height == m_height;

	bool reinit = false;
	if(!m_isInit || m_useCuda != cudaImage){
		m_isInit = init_decoder(info.width, info.height, FAST_RGB8, cudaImage);
		m_useCuda = cudaImage;
		reinit = true;
	}

	if(ret != FAST_OK)
        return false;

    ret = fastJpegDecode(m_handle, &info);

    if(ret == FAST_OK){
		if(cudaImage){
			if(reinit || !output.get() || output->width != info.width || output->height != info.height){
                                output.reset(new RTSPImage(info.width, info.height, RTSPImage::CUDA_RGB));
			}
		}else{
			if(reinit || !output.get() || output->width != info.width || output->height != info.height){
                                output.reset(new RTSPImage(info.width, info.height, RTSPImage::RGB));
			}
		}

        int pitch = output->width * 3;

		if(cudaImage){
			ret = fastExportToDeviceCopy(m_DeviceToDevice, output->cudaRgb, output->width, pitch,
										 output->height, &params);
		}else{
			ret = fastExportToHostCopy(m_DeviceToHost, output->rgb.data(), output->width, pitch,
									   output->height, &params);
		}
		return ret == FAST_OK;
    }
    return false;
}
