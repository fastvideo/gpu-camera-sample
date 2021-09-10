#include "common.h"

#include <cuda_runtime_api.h>

RTSPImage::RTSPImage(){

}

RTSPImage::RTSPImage(int w, int h, RTSPImage::TYPE tp){
	if(tp == YUV)
		setYUV(w, h);
	else if(tp == NV12)
		setNV12(w, h);
    else if(tp == P010)
        setP010(w, h);
    else if(tp == RGB)
		setRGB(w, h);
	else if(tp == GRAY)
		setGray(w, h);
	else if(tp == CUDA_RGB){
		setCudaRgb(w, h);
	}else if(tp == CUDA_GRAY){
		setCudaGray(w, h);
	}
}

RTSPImage::RTSPImage(const RTSPImage &o){
	width = o.width;
	height = o.height;
	type = o.type;
	rgb = o.rgb;
	yuv = o.yuv;
}

RTSPImage::~RTSPImage(){
	releaseCudaRgbBuffer();
}

void RTSPImage::setYUV(uint8_t *data[], int linesize[], int w, int h)
{
	type = YUV;
	width = w;
	height = h;
    size_t size1 = static_cast<size_t>(w * h);
    size_t size2 = static_cast<size_t>(w/2 * h/2);
	//size_t size3 = static_cast<size_t>(linesize[2] * h/2);
	yuv.resize(size1 + size2 * 2);

    int l1 = linesize[0];
    int l2 = linesize[1];
    int l3 = linesize[2];

    for(int i = 0; i < h; ++i){
        uchar *dY = data[0] + i * l1;
        uchar *dOY = yuv.data() + i * w;
        std::copy(dY, dY + w, dOY);
    }
    uchar *offU = yuv.data() + size1;
    uchar *offV = yuv.data() + size1 + size2;
    for(int i = 0; i < h/2; ++i){
        uchar *dU = data[1] + i * l2;
        uchar *dOU = offU + i * w/2;
        std::copy(dU, dU + w/2, dOU);
        uchar *dV = data[2] + i * l3;
        uchar *dOV = offV + i * w/2;
        std::copy(dV, dV + w/2, dOV);
    }

//	std::copy(data[0], data[0] + size1, yuv.data());
//	std::copy(data[1], data[1] + size2, yuv.data() + size1);
//	std::copy(data[2], data[2] + size2, yuv.data() + size1 + size2);
}

void RTSPImage::setNV12(uint8_t *data[], int linesize[], int w, int h)
{
	type = NV12;
	width = w;
	height = h;
    size_t size1 = static_cast<size_t>(w * h);
    size_t size2 = static_cast<size_t>(w * h/2);
	//size_t size3 = static_cast<size_t>(linesize[1]/2 * h/2);
	yuv.resize(size1 + size2 * 2);

    int l1 = linesize[0];
    int l2 = linesize[1];

    for(int i = 0; i < h; ++i){
        uchar *dY = data[0] + i * l1;
        uchar *dOY = yuv.data() + i * w;
        std::copy(dY, dY + w, dOY);
    }
    uchar *offUV = yuv.data() + size1;
    for(int i = 0; i < h/2; ++i){
        uchar *dUV = data[1] + i * l2;
        uchar *dOUV = offUV + i * w;
        std::copy(dUV, dUV + w, dOUV);
    }

    //std::copy(data[0], data[0] + size1, yuv.data());
    //std::copy(data[1], data[1] + size2 * 2, yuv.data() + size1);
}

void RTSPImage::setP010(uint8_t *data[], int linesize[], int w, int h)
{
    type = P010;
    width = w;
    height = h;
    size_t size1 = static_cast<size_t>(w * h * 2);
    size_t size2 = static_cast<size_t>(w * h/2 * 2);
    //size_t size3 = static_cast<size_t>(linesize[1]/2 * h/2);
    yuv.resize(size1 + size2 * 2);

    int l1 = linesize[0];
    int l2 = linesize[1];

    int bpl = w * 2;

    for(int i = 0; i < h; ++i){
        uchar *dY = data[0] + i * l1;
        uchar *dOY = yuv.data() + i * bpl;
        std::copy(dY, dY + bpl, dOY);
    }
    uchar *offUV = yuv.data() + size1;
    for(int i = 0; i < h/2; ++i){
        uchar *dUV = data[1] + i * l2;
        uchar *dOUV = offUV + i * bpl;
        std::copy(dUV, dUV + bpl, dOUV);
    }
}

bool RTSPImage::setCudaRgb(int w, int h){
	if(cudaRgb && w != width && h != height)
		releaseCudaRgbBuffer();
	type = CUDA_RGB;
	width = w;
    height = h;
    size_t sz = w * h * 3;
	cudaSize = sz;
	return cudaMalloc(&cudaRgb, sz) == cudaSuccess;
}

bool RTSPImage::setCudaGray(int w, int h)
{
	if(cudaRgb && w != width && h != height)
		releaseCudaRgbBuffer();
	type = CUDA_GRAY;
	width = w;
    height = h;
    size_t sz = w * h;
	cudaSize = sz;
	return cudaMalloc(&cudaRgb, sz) == cudaSuccess;
}

void RTSPImage::releaseCudaRgbBuffer(){
	if(cudaRgb){
		cudaFree(cudaRgb);
		cudaRgb = nullptr;
	}
	width = height = 0;
	cudaSize = 0;
}

void RTSPImage::setYUV(int w, int h){
	type = YUV;
	width = w;
	height = h;
	yuv.resize(w * h + w/2 * h/2 * 2);
}

void RTSPImage::setNV12(int w, int h){
    type = NV12;
	width = w;
	height = h;
    yuv.resize(w * h + w/2 * h/2 * 2);
}

void RTSPImage::setP010(int w, int h)
{
    type = P010;
    width = w;
    height = h;
    yuv.resize(w * h * 2 + w/2 * h/2 * 2 * 2);
}

void RTSPImage::setRGB(int w, int h){
	type = RGB;
	width = w;
	height = h;
	rgb.resize(w * h * 3);
}

void RTSPImage::setGray(int w, int h){
	type = GRAY;
	width = w;
	height = h;
	rgb.resize(w * h);
}

bool RTSPImage::empty() const{
	return width == 0 || height == 0;
}
