#ifndef CUDAPROCESSORGRAY_H
#define CUDAPROCESSORGRAY_H
#include "CUDAProcessorBase.h"

class CUDAProcessorGray : public CUDAProcessorBase
{
public:
    CUDAProcessorGray(QObject *parent = nullptr);
    ~CUDAProcessorGray() override;
    virtual fastStatus_t Init(CUDAProcessorOptions& options);
    virtual fastStatus_t Transform(ImageT *image, CUDAProcessorOptions& opts) override;
    virtual bool isGrayscale() override {return true;}
    virtual void freeFilters() override;
    virtual fastStatus_t export8bitData(void* dstPtr, bool forceRGB = true) override;

private:
    fastSurfaceConverterHandle_t    hGrayToRGBTransform = nullptr;
    fastDeviceSurfaceBufferHandle_t dstGrayBuffer = nullptr;
    fastExportToHostHandle_t        hDeviceToHostGrayAdapter = nullptr;
};

#endif // CUDAPROCESSORGRAY_H
