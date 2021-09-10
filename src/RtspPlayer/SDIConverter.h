#ifndef SDICONVERTER_H
#define SDICONVERTER_H

#include "common.h"

#include <fastvideo_sdk.h>

class SDIConverter
{
public:
	SDIConverter();
	~SDIConverter();
	/**
	 * @brief convertToRgb
	 * @param image
	 * @param cudaRgb
	 */
	bool convertToRgb(PImage image, void* cudaRgb);
	/**
	 * @brief release
	 */
	void release();

private:
	fastSDIImportFromHostHandle_t m_hImport = nullptr;
	fastDeviceSurfaceBufferHandle_t srcBuffer = nullptr;
	fastDeviceSurfaceBufferHandle_t dstBuffer = nullptr;
	fastExportToDeviceHandle_t m_hExport = nullptr;
    fastSurfaceConverterHandle_t mSurfaceConverter = nullptr;
	fastSurfaceFormat_t surfaceFmt;

	bool initSdiConvert(PImage image);
	void releaseSdiConvert();
	bool sdiConvert(PImage image, void *cudaRgb);

	void convertNv12ToYuv420(PImage image);

	int m_prevWidth = 0;
	int m_prevHeight = 0;
    RTSPImage::TYPE m_prevType;

	bytearray m_Yuv;
};

#endif // SDICONVERTER_H
