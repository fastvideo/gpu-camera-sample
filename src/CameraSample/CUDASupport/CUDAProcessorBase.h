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

#ifndef CUDAPROCESSORBASE_H
#define CUDAPROCESSORBASE_H

#include "CUDAProcessorOptions.h"
#include <list>
#include <memory>

#include "FastAllocator.h"

#include "fastvideo_sdk.h"
#include "fastvideo_mjpeg.h"
#include "fastvideo_denoise.h"
#include "fastvideo_nppFilter.h"
#include "fastvideo_nppResize.h"
#include "fastvideo_nppGeometry.h"

#include <QVector>


#include <QObject>
#include <QVector>
#include <QMap>
#include <QMutex>
#include <QPair>
#include <QThread>
#include <QSharedMemory>
#include <QApplication>
#include <QDesktopWidget>
#include <QDir>
#include <QFileInfo>
#include <QDateTime>
#include <QElapsedTimer>
#include <QDebug>
#include <iostream>

#include "timing.hpp"
#include "helper_image/helper_ppm.hpp"
#include "Image.h"
#include "helper_jpeg.hpp"
#include "FrameBuffer.h"

#include "Globals.h"

#include <cuda_runtime.h>
#include <cuda.h>


class CUDAProcessorBase : public QObject
{
    Q_OBJECT
public:
    CUDAProcessorBase(QObject* parent = nullptr);
    ~CUDAProcessorBase() override;

    virtual fastStatus_t Init(CUDAProcessorOptions & options);
    virtual fastStatus_t InitFailed(const char *errStr, fastStatus_t ret);

    virtual fastStatus_t Transform(GPUImage_t *image, CUDAProcessorOptions& opts);
    virtual fastStatus_t TransformFailed(const char *errStr, fastStatus_t ret, fastGpuTimerHandle_t profileTimer);

    virtual fastStatus_t Close();
    virtual void         freeFilters();

    virtual fastStatus_t export8bitData(void* dstPtr, bool forceRGB = true);
    virtual fastStatus_t exportJPEGData(void* dstPtr, unsigned jpegQuality, unsigned &size);
    virtual fastStatus_t exportNV12Data(void* dstPtr);
    virtual fastStatus_t exportP010Data(void* dstPtr);
    virtual fastStatus_t exportYuv8Data(void* dstPtr);
    virtual fastStatus_t exportNV12DataDevice(void* dstPtr);
    virtual fastStatus_t exportP010DataDevice(void* dstPtr);
    virtual fastStatus_t exportYuv8DataDevice(void* dstPtr);

    fastStatus_t exportRawData(void* dstPtr, unsigned int &w, unsigned int &h, unsigned int &pitch);
    fastStatus_t export16bitData(void* dstPtr, unsigned int &w, unsigned int &h, unsigned int &pitch);
    fastStatus_t exportLinearizedRaw(void* dstPtr, unsigned int &w, unsigned int &h, unsigned int &pitch);

    fastSurfaceFormat_t getInputSurfaceFmt();
    QSize        getMaxInputSize();
    void cudaMemoryInfo(const char *str);
    void clearExifSections();

    virtual void* GetFrameBuffer();

    virtual bool isGrayscale(){return false;}
    QString getLastErrorDescription(){ return mErrString; }
    fastStatus_t getLastError(){ return mLastError; }
    bool isInitialized(){return mInitialised;}
    void setInfo(bool info)
    {
        QMutexLocker lock(&mut);
        this->info = info;
    }

    bool haveCUDAJpeg()
    {
        return (hJpegEncoder != nullptr);
    }


    fastBayerPattern_t       BayerFormat;
    QMutex                   mut;
    QMutex                   mut2;
    QMap<QString, float>     stats;
    QMap<QString, float>     stats2;
    fastExportToHostHandle_t hBitmapExport = nullptr;
    fastLut_16_t             outLut;

protected:
    bool         info = true;

    static const int JPEG_HEADER_SIZE = 1024;
    static const int FRAME_TIME = 2;

    fastSurfaceFormat_t surfaceFmt {};
    bool                mInitialised = false;
    QString             mErrString;
    fastStatus_t        mLastError {};

    fastImportFromDeviceHandle_t    hDeviceToDeviceAdapter = nullptr;
    fastDeviceSurfaceBufferHandle_t srcBuffer = nullptr;

    fastRawUnpackerHandle_t         hRawUnpacker = nullptr;

    fastImageFiltersHandle_t        hSam = nullptr;
    fastDeviceSurfaceBufferHandle_t samBuffer = nullptr;
    fastMuxHandle_t                 hSamMux = nullptr;
    fastDeviceSurfaceBufferHandle_t samMuxBuffer = nullptr;

    fastImageFiltersHandle_t        hBpc = nullptr;
    fastDeviceSurfaceBufferHandle_t bpcBuffer = nullptr;
    fastMuxHandle_t                 hBpcMux = nullptr;
    fastDeviceSurfaceBufferHandle_t bpcMuxBuffer = nullptr;

    fastImageFiltersHandle_t        hLinearizationLut = nullptr;
    fastDeviceSurfaceBufferHandle_t linearizationLutBuffer = nullptr;

    fastImageFiltersHandle_t        hWhiteBalance = nullptr;
    fastDeviceSurfaceBufferHandle_t whiteBalanceBuffer = nullptr;

    fastDebayerHandle_t             hDebayer = nullptr;
    fastDeviceSurfaceBufferHandle_t debayerBuffer = nullptr;

    //Output lut (gamma)
    fastImageFiltersHandle_t        hOutLut = nullptr;
    fastDeviceSurfaceBufferHandle_t outLutBuffer = nullptr;

    fastSurfaceConverterHandle_t    h16to8Transform = nullptr;
    fastDeviceSurfaceBufferHandle_t displaytBuffer = nullptr;

    fastSDIExportToHostHandle_t     hSdiExportToHost = nullptr;
    fastSDIExportToHostHandle_t     hSdiExportToHost10bit = nullptr;
    fastSDIExportToHostHandle_t     hSdiExportToHostYuv8bit = nullptr;

    fastSDIExportToDeviceHandle_t   hSdiExportToDevice = nullptr;
    fastSDIExportToDeviceHandle_t   hSdiExportToDevice10bit = nullptr;
    fastSDIExportToDeviceHandle_t   hSdiExportToDeviceYuv8bit = nullptr;

    fastDeviceSurfaceBufferHandle_t dstBuffer = nullptr;

    //Motion Jpeg stuff
    std::unique_ptr<unsigned char, FastAllocator> hJpegStream;
    fastJpegEncoderHandle_t hJpegEncoder = nullptr;
    fastJfifInfo_t          jfifInfo{};
    unsigned int            jpegStreamSize;

    //Denoise stuff
    fastDenoiseHandle_t             hDenoise = nullptr;
    fastDeviceSurfaceBufferHandle_t denoiseBuffer = nullptr;
    denoise_static_parameters_t     denoiseParameters {};
    fastMuxHandle_t                 hDenoiseMux = nullptr;
    fastDeviceSurfaceBufferHandle_t denoiseMuxBuffer = nullptr;


    fastExportToHostHandle_t hDeviceToHostRawAdapter = nullptr;
    fastExportToHostHandle_t hDeviceToHostLinRawAdapter = nullptr;
    fastExportToHostHandle_t hDeviceToHostAdapter = nullptr;
    fastExportToHostHandle_t hDeviceToHost16Adapter = nullptr;


    //OpenGL stuff
    void*                      hGLBuffer = nullptr;
    fastExportToDeviceHandle_t hExportToDevice = nullptr;

    template<typename T>
    void InitLut(T & param, unsigned short blackLevel, double scale, const QVector<unsigned short> & linearizationLut = QVector<unsigned short>());

signals:
    void initialized(const QString& info);
    void finished();
    void error();
protected:

};
template<typename T>
void CUDAProcessorBase::InitLut(T & param, unsigned short blackLevel, double scale, const QVector<unsigned short> & linearizationLut)
{
    if(linearizationLut.empty())
    {
        int i = 0;
        for(auto& l : param.lut)
        {
            l = static_cast<unsigned short>(qBound<double>(0, (i - blackLevel)  * scale, 1) * 65535);
            i++;
        }
    }
    else
    {
        auto itr = linearizationLut.begin();
        for(auto & l : param.lut)
        {
            if( itr != linearizationLut.end() )
            {
                l = static_cast<unsigned short>(qBound<double>(0, (*itr - blackLevel)  * scale, 1) * 65535);
                ++itr;
                continue;
            }
            break;
        }
    }
}

Q_DECLARE_TYPEINFO(fastJpegExifSection_t, Q_COMPLEX_TYPE);


#endif // CUDAPROCESSORBASE_H
