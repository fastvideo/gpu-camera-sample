#ifndef FASTVIDEO_DECODER_H
#define FASTVIDEO_DECODER_H

#include <fastvideo_sdk.h>

#include "common.h"
#include "common_utils.h"

class fastvideo_decoder
{
public:
    fastvideo_decoder();
    ~fastvideo_decoder();

	bool init_decoder(uint32_t width, uint32_t height, fastSurfaceFormat_t fmt, bool cudaImage = false);
	void release_decoder();

	bool decode(const uint8_t *input, uint32_t len, PImage &output, bool cudaImage = false);
	bool decode(const bytearray& input, PImage &output, bool cudaImage = false);

private:
    fastJpegDecoderHandle_t m_handle = nullptr;
    fastDeviceSurfaceBufferHandle_t m_dHandle = nullptr;
    fastSurfaceFormat_t m_surfaceFmt;
    fastExportToHostHandle_t m_DeviceToHost = nullptr;
	fastExportToDeviceHandle_t m_DeviceToDevice = nullptr;
	bool m_useCuda = false;
    bool m_isInit = false;
	unsigned m_width = 0;
	unsigned m_height = 0;

    bytearray buf;
};

#endif // FASTVIDEO_DECODER_H
