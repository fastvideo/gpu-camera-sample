#include "SDIConverter.h"

SDIConverter::SDIConverter()
{
	fastStatus_t ret;
	ret = fastInit(1, false);
}

SDIConverter::~SDIConverter()
{
	release();
}

bool SDIConverter::convertToRgb(PImage image, void *cudaRgb)
{
	initSdiConvert(image);

	bool res = sdiConvert(image, cudaRgb);

	m_prevWidth = image->width;
	m_prevHeight = image->height;
	m_prevType = image->type;

	return res;
}

void SDIConverter::release()
{
	releaseSdiConvert();
}

#define CHECK_FAST(val)

unsigned GetPitchFromSurface(fastSurfaceFormat_t fmt, int width)
{
	switch (fmt) {
		case FAST_RGB12:
			return width * 2 * 3;
		case FAST_RGB8:
		case FAST_BGR8:
			return width * 3;
		case FAST_RGB16:
			return width * 3 * 2;
		case FAST_I8:
			return width;
		case FAST_I10:
		case FAST_I12:
		case FAST_I14:
		case FAST_I16:
			return width * 2;
	}
	return width * 3;
}

bool SDIConverter::initSdiConvert(PImage image)
{
	if(m_hImport && m_hExport && image->width == m_prevWidth && image->height == m_prevHeight){
		return true;
	}

	releaseSdiConvert();

	unsigned width = image->width;
	unsigned height = image->height;

	fastStatus_t res;

	fastSDIFormat_t SDIFormat;

	if(image->type == Image::YUV){
		SDIFormat = FAST_SDI_NV12_BT601;
	}else if(image->type == Image::NV12){
		SDIFormat = FAST_SDI_NV12_BT601;
	}

	res = (fastSDIImportFromHostCreate(
		&m_hImport,

		SDIFormat,

		width,
		height,

		&srcBuffer
	));

	surfaceFmt = FAST_RGB8;

	res = (fastExportToDeviceCreate(
		&m_hExport,

		&surfaceFmt,

		srcBuffer
	));

	unsigned requestedMemSpace = 0;
	unsigned tmp = 0;

	if( m_hImport != nullptr )
	{
		res = (fastSDIImportFromHostGetAllocatedGpuMemorySize(m_hImport, &tmp));
		requestedMemSpace += tmp;
	}

	if( m_hExport != nullptr )
	{
		res = (fastExportToDeviceGetAllocatedGpuMemorySize(m_hExport, &tmp));
		requestedMemSpace += tmp;
	}

	return res == FAST_OK;
}

void SDIConverter::releaseSdiConvert()
{
	if(m_hImport){
		fastSDIImportFromHostDestroy(m_hImport);
		m_hImport = nullptr;
	}
	if(m_hExport){
		fastExportToDeviceDestroy(m_hExport);
		m_hExport = nullptr;
	}
}

bool SDIConverter::sdiConvert(PImage image, void *cudaRgb)
{
	unsigned width = image->width;
	unsigned height = image->height;

	fastStatus_t res;

	if(image->type == Image::NV12){
		res = (fastSDIImportFromHostCopy(
			m_hImport,

			image->yuv.data(),

			width,
			height
		));
	}else{
		convertNv12ToYuv420(image);
		res = (fastSDIImportFromHostCopy(
			m_hImport,

			m_Yuv.data(),

			width,
			height
		));
	}

	if(res != FAST_OK){
		qDebug("Error fastSDIImportFromHostCopy %d", res);
		return false;
	}

	bool convertToBGR = false;

	fastExportParameters_t exportParameters;
    exportParameters.convert = convertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;
	res = (fastExportToDeviceCopy(
		m_hExport,
		cudaRgb,
		width,
        width * 3,
		height,

		&exportParameters
	));

	if(res != FAST_OK){
		qDebug("Error fastExportToDeviceCopy %d", res);
		return false;
	}

	return res == FAST_OK;
}

void SDIConverter::convertNv12ToYuv420(PImage image)
{
	int w = image->width;
	int h = image->height;
	size_t size1 = static_cast<size_t>(w * h);
	size_t size2 = static_cast<size_t>(w/2 * h/2);
	//size_t size3 = static_cast<size_t>(w/2 * h/2);
	m_Yuv.resize(image->yuv.size());

	uchar* data = image->yuv.data();

	std::copy(data, data + size1, m_Yuv.data());

	uchar *U = data + size1;
	uchar *V = data + size1 + size2;

	uchar *data1 = m_Yuv.data() + size1;

	for(size_t x = 0; x < size2; ++x){
		data1[x * 2 + 0] = U[x];
		data1[x * 2 + 1] = V[x];
	}
}
