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

#include "vutils.h"

#include "JpegEncoder.h"

void encodeJpeg(int idthread, unsigned char* data, int width, int height, int channels, Buffer& output)
{
#if 0
	QImage::Format fmt = QImage::Format_Grayscale8;
	if(channels == 3)
		fmt = QImage::Format_RGB888;
	QImage img(data, width, height, fmt);

	QByteArray d;
	QDataStream stream(&d, QIODevice::WriteOnly);

	QImageWriter writer(stream.device(), "jpeg");
	writer.setQuality(40);

	writer.write(img);
	output.buffer.resize(d.size());
	output.size = d.size();
	std::copy(d.data(), d.data() + d.size(), output.buffer.data());
#else
    idthread;
	jpeg_encoder enc;
	enc.encode(data, width, height, channels, output.buffer, 30);
	output.size = output.buffer.size();
#endif
}

void copyPartImage(unsigned char *input, size_t xoff, size_t yoff, size_t sizeEl, size_t linesize,
				   size_t hpart, size_t linesizepart, unsigned char *output)
{
	for(size_t y = 0; y < hpart; ++y){
		unsigned char *d = &input[(yoff + y) * linesize + xoff * sizeEl];
		unsigned char *o = &output[y * linesizepart];
		std::copy(d, d + linesizepart, o);
	}
}

/////////////////////////

int set_hwframe_ctx(AVCodecContext *ctx, AVBufferRef *hw_device_ctx, int width, int height, AVPixelFormat pixfmt)
{
    AVBufferRef *hw_frames_ref;
    AVHWFramesContext *frames_ctx = NULL;
    int err = 0;

    if (!(hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx))) {
        fprintf(stderr, "Failed to create VAAPI frame context.\n");
        return -1;
    }
    frames_ctx = (AVHWFramesContext *)(hw_frames_ref->data);
    frames_ctx->format    = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = pixfmt;
    frames_ctx->width     = width;
    frames_ctx->height    = height;
    frames_ctx->initial_pool_size = 20;
    if ((err = av_hwframe_ctx_init(hw_frames_ref)) < 0) {
        printf("Failed to initialize CUDA frame context. Error code: %d\n", err);
        av_buffer_unref(&hw_frames_ref);
        return err;
    }
    ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ref);
    if (!ctx->hw_frames_ctx)
        err = AVERROR(ENOMEM);

    av_buffer_unref(&hw_frames_ref);
    return err;
}
