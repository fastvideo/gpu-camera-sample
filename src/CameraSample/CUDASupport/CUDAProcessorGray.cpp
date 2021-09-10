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

#include "CUDAProcessorGray.h"

#ifdef STATIC_BUILD
extern "C"  fastStatus_t fastEnableWatermark(bool isEnabled);
#endif



CUDAProcessorGray::CUDAProcessorGray(QObject* parent) : CUDAProcessorBase(parent)
{
    hGrayToRGBTransform = nullptr;
    dstGrayBuffer = nullptr;
    hDeviceToHostGrayAdapter = nullptr;
}

CUDAProcessorGray::~CUDAProcessorGray()
{
    freeFilters();
}

void CUDAProcessorGray::freeFilters()
{
    CUDAProcessorBase::freeFilters();
    if(hGrayToRGBTransform != nullptr)
    {
        fastSurfaceConverterDestroy( hGrayToRGBTransform );
        hGrayToRGBTransform = nullptr;
        cudaMemoryInfo("Destroyed hGrayToRGBTransform");
    }
    if(hDeviceToHostGrayAdapter != nullptr)
    {
        fastExportToHostDestroy( hDeviceToHostGrayAdapter );
        hDeviceToHostGrayAdapter = nullptr;
        cudaMemoryInfo("Destroyed hDeviceToHostGrayAdapter");
    }
}

fastStatus_t CUDAProcessorGray::Init(CUDAProcessorOptions &options)
{
    fastStatus_t ret;
    FastAllocator alloc;

    if(mInitialised)
    {
        mInitialised = false;
        freeFilters();
    }

    if(info)
        qDebug("Initialising CUDAProcessorBase...");

    mut.lock();

    mLastError = FAST_OK;
    mErrString = QString();

    ret = fastInit(1U << 0, false);
    if(ret != FAST_OK)
        InitFailed("fastInit failed", ret);

    fastSdkParametersHandle_t handle = nullptr;
    ret = fastGetSdkParametersHandle(&handle);
    ret = fastDenoiseLibraryInit(handle);

    if(Globals::gEnableLog)
    {
        fastTraceCreate(QDir::toNativeSeparators(
                            QStringLiteral("%1.log").arg(
                                QDateTime::currentDateTime().toString(
                                QStringLiteral("dd_MM_yyyy_hh_mm_ss")))).
                                toStdString().c_str());
        fastTraceEnableLUTDump(false);
        fastTraceEnableFlush(true);
        fastEnableInterfaceSynchronization(true);
    }

    stats[QStringLiteral("inputWidth")] = -1;
    stats[QStringLiteral("inputHeight")] = -1;

    fastSurfaceFormat_t srcSurfaceFmt  = options.SurfaceFmt;

    unsigned int maxWidth = options.MaxWidth;
    unsigned int maxHeight = options.MaxHeight;

    fastDeviceSurfaceBufferHandle_t* bufferPtr = nullptr;

    if(options.Packed)
    {
        fastSDIRaw12Import_t p = {false};
        ret = fastRawImportFromDeviceCreate(
                    &hRawUnpacker,

                    FAST_RAW_XIMEA12,

                    &p,
                    maxWidth,
                    maxHeight,

                    &srcBuffer
                    );

        if(ret != FAST_OK)
            return InitFailed("fastRawUnpackerCreate failed",ret);

        cudaMemoryInfo("Created fastRawUnpackerCreate");
    }
    else
    {
        ret = fastImportFromDeviceCreate(
                    &hDeviceToDeviceAdapter,

                    srcSurfaceFmt,
                    maxWidth,
                    maxHeight,

                    &srcBuffer
                    );
        //
        if(ret != FAST_OK)
            return InitFailed("fastImportFromHostCreate failed",ret);
        //
        cudaMemoryInfo("Created hHostToDeviceAdapter");
    }
    bufferPtr = &srcBuffer;

    //Raw data export
    ret = fastExportToHostCreate(
                &hDeviceToHostRawAdapter,
                &srcSurfaceFmt,
                *bufferPtr
                );

    if(ret != FAST_OK)
        return InitFailed("fastExportToHostCreate for Raw data failed",ret);

    cudaMemoryInfo("Created hDeviceToHostRawAdapter");

    //SAM
    if(options.SurfaceFmt == FAST_I8)
    {
        fastSam_t samParameter;
        samParameter.correctionMatrix = options.MatrixA;
        samParameter.blackShiftMatrix = (char*)options.MatrixB;
        ret = fastImageFilterCreate(
               &hSam,

               FAST_SAM,
               &samParameter,

               maxWidth,
               maxHeight,

               *bufferPtr,
               &samBuffer
               );
    }
    else
    {
        fastSam16_t samParameter;
        samParameter.correctionMatrix = options.MatrixA;
        samParameter.blackShiftMatrix = (short*)options.MatrixB;
        ret = fastImageFilterCreate(
               &hSam,

               FAST_SAM16,
               &samParameter,

               maxWidth,
               maxHeight,

               *bufferPtr,
               &samBuffer
               );
    }
    if(ret != FAST_OK)
        return InitFailed("fastImageFilterCreate for MAD failed",ret);

    cudaMemoryInfo("Created SAM");

    if(samBuffer)
    {
        fastDeviceSurfaceBufferHandle_t srcBuffers[2] = {*bufferPtr, samBuffer};

        ret = fastMuxCreate(
                    &hSamMux,
                    srcBuffers,
                    2,
                    &samMuxBuffer
                    );
    }

    bufferPtr = &samMuxBuffer;

    if(ret != FAST_OK)
        return InitFailed("fastMuxCreate hSamMux failed",ret);
    cudaMemoryInfo("Created hSamMux");

    unsigned short whiteLevel = options.WhiteLevel;
    unsigned short blackLevel = options.BlackLevel;
    double scale = 1. / (double(whiteLevel - blackLevel));
    if(options.SurfaceFmt == FAST_I8)
    {
        fastLut_8_16_t lutParameter;
        InitLut<fastLut_8_16_t>(lutParameter, blackLevel, scale, options.LinearizationLut);
        ret = fastImageFilterCreate(
                    &hLinearizationLut,
                    FAST_LUT_8_16,
                    &lutParameter,

                    maxWidth,
                    maxHeight,

                    *bufferPtr,
                    &linearizationLutBuffer
                    );
    }
    else if(options.SurfaceFmt == FAST_I10)
    {
        fastLut_10_t lutParameter;
        InitLut<fastLut_10_t>(lutParameter, blackLevel, scale, options.LinearizationLut);
        ret = fastImageFilterCreate(
                    &hLinearizationLut,
                    FAST_LUT_10_16,
                    &lutParameter,

                    maxWidth,
                    maxHeight,

                    *bufferPtr,
                    &linearizationLutBuffer
                    );
    }
    else if(options.SurfaceFmt == FAST_I12)
    {
        fastLut_12_t lutParameter;
        InitLut<fastLut_12_t>(lutParameter, blackLevel, scale, options.LinearizationLut);
        ret = fastImageFilterCreate(
                    &hLinearizationLut,
                    FAST_LUT_12_16,
                    &lutParameter,

                    maxWidth,
                    maxHeight,

                    *bufferPtr,
                    &linearizationLutBuffer
                    );
    }
    else if(options.SurfaceFmt == FAST_I14)
    {
        fastLut_16_t lutParameter;
        InitLut<fastLut_16_t>(lutParameter, blackLevel, scale, options.LinearizationLut);
        ret = fastImageFilterCreate(
                    &hLinearizationLut,
                    FAST_LUT_14_16,
                    &lutParameter,

                    maxWidth,
                    maxHeight,

                    *bufferPtr,
                    &linearizationLutBuffer
                    );
    }
    else
    {
        fastLut_16_t lutParameter;
        InitLut<fastLut_16_t>(lutParameter,blackLevel,scale,options.LinearizationLut);
        ret = fastImageFilterCreate(
                    &hLinearizationLut,

                    FAST_LUT_16_16,
                    &lutParameter,

                    maxWidth,
                    maxHeight,

                    *bufferPtr,
                    &linearizationLutBuffer
                    );
    }
    //
    if(ret != FAST_OK)
        return InitFailed("fastImageFilterCreate for Linearization Lut failed",ret);
    //
    cudaMemoryInfo("Created hLinearizationLut");
    bufferPtr = &linearizationLutBuffer;

    srcSurfaceFmt = FAST_I16;


    //BPC
    fastBadPixelCorrection_t bpcParameter;
    bpcParameter.pattern = options.BayerFormat;

    ret = fastImageFilterCreate(
               &hBpc,

               FAST_BAD_PIXEL_CORRECTION_5X5,
               &bpcParameter,

               maxWidth,
               maxHeight,

               *bufferPtr,
               &bpcBuffer
               );

    if(ret != FAST_OK)
        return InitFailed("fastImageFilterCreate for MAD failed",ret);

    cudaMemoryInfo("Created BPC");

    if(*bufferPtr && bpcBuffer)
    {
        fastDeviceSurfaceBufferHandle_t srcBuffers[2] = {*bufferPtr, bpcBuffer};

        ret = fastMuxCreate(
                    &hBpcMux,
                    srcBuffers,
                    2,
                    &bpcMuxBuffer
                    );
    }

    bufferPtr = &bpcMuxBuffer;

    if(ret != FAST_OK)
        return InitFailed("fastMuxCreate hBpcMux failed",ret);
    cudaMemoryInfo("Created hBpcMux");

    //Denoise
    if( true )
    {
        denoise_static_parameters_t denoiseParameters;
        memcpy(&denoiseParameters, &options.DenoiseStaticParams, sizeof(denoise_static_parameters_t));

        ret = fastDenoiseCreate(
                    &hDenoise,

                    FAST_I16,
                    &denoiseParameters,
                    maxWidth,
                    maxHeight,

                    *bufferPtr,
                    &denoiseBuffer
                    );

        bufferPtr = &denoiseBuffer;

        if(ret != FAST_OK)
            return InitFailed("fastDenoiseCreate failed",ret);
        cudaMemoryInfo("Created hDenoise");

        //Denoise mux
        if(bpcMuxBuffer && denoiseBuffer)
        {
            fastDeviceSurfaceBufferHandle_t srcBuffers[2] = {bpcMuxBuffer, denoiseBuffer};

            ret = fastMuxCreate(
                        &hDenoiseMux,
                        srcBuffers,
                        2,
                        &denoiseMuxBuffer
                        );

            bufferPtr = &denoiseMuxBuffer;

            if(ret != FAST_OK)
                return InitFailed("fastMuxCreate failed",ret);
            cudaMemoryInfo("Created hDenoiseMux");
        }
    }

    //Output LUT
    for(int i = 0; i < 16384; i++)
        outLut.lut[i] = static_cast<unsigned short>(i * 4);//rgbLut[0][i];
    ret = fastImageFilterCreate(
                &hOutLut,

                FAST_LUT_16_16,
                &outLut,

                maxWidth,
                maxHeight,

                *bufferPtr,
                &outLutBuffer
                );

    if(ret != FAST_OK)
        return InitFailed("fastImageFilterCreate for Output lut failed",ret);

    cudaMemoryInfo("Created hOutLut");
    bufferPtr = &outLutBuffer;


    //16 bit gray data export
    ret = fastExportToHostCreate(
                &hDeviceToHost16Adapter,
                &srcSurfaceFmt,
                *bufferPtr
                );

    if(ret != FAST_OK)
        return InitFailed("fastExportToHostCreate for 16 bit data failed",ret);

    cudaMemoryInfo("Created hDeviceToHost16Adapter");

    if(srcSurfaceFmt != FAST_I16 && info)
        qDebug("hDeviceToHost16Adapter returned invalid format = %u", srcSurfaceFmt);


    //16 to 8 bit transform
    fastBitDepthConverter_t conv;
    conv.targetBitsPerChannel = 8;
    conv.isOverrideSourceBitsPerChannel = false;
    ret = ( fastSurfaceConverterCreate(
                &h16to8Transform,
                FAST_BIT_DEPTH,

                &conv,

                maxWidth,
                maxHeight,

                *bufferPtr,
                &dstGrayBuffer
                ) );

    bufferPtr = &dstGrayBuffer;

    if(ret != FAST_OK)
        return InitFailed("h16to8Transform failed",ret);

    cudaMemoryInfo("Created h16to8Transform");

    //Export to host gray 8 bit image
    ret = ( fastExportToHostCreate(
                &hDeviceToHostGrayAdapter,
                &srcSurfaceFmt,
                *bufferPtr
                ) );

    if(ret != FAST_OK)
        return InitFailed("hDeviceToHostGrayAdapter failed",ret);

    cudaMemoryInfo("Created hDeviceToHostGrayAdapter");
    if(srcSurfaceFmt != FAST_I8 && info)
        qDebug("fastExportToHostCreate returned invalid format = %u", srcSurfaceFmt);

    //Gray 8 bit to RGB 8 bit
    ret = ( fastSurfaceConverterCreate(
                &hGrayToRGBTransform,
                FAST_GRAYSCALE_TO_RGB,

                &conv,

                maxWidth,
                maxHeight,

                *bufferPtr,
                &dstBuffer
                ) );

    bufferPtr = &dstBuffer;

    if(ret != FAST_OK)
        return InitFailed("h16to8Transform failed",ret);

    cudaMemoryInfo("Created h16to8Transform");

    if(srcSurfaceFmt != FAST_I8 && info)
        qDebug("fastExportToHostCreate returned invalid format = %u", srcSurfaceFmt);

    //Export to host rgb image
    ret = ( fastExportToHostCreate(
                &hDeviceToHostAdapter,
                &srcSurfaceFmt,
                *bufferPtr
                ) );

    if(ret != FAST_OK)
        return InitFailed("fastExportToHostCreate failed",ret);

    cudaMemoryInfo("Created hDeviceToHostAdapter");

    if(srcSurfaceFmt != FAST_RGB8 && info)
        qDebug("fastExportToHostCreate returned invalid format = %u", srcSurfaceFmt);

    //Open GL
    ret = fastExportToDeviceCreate(
        &hExportToDevice,
        &srcSurfaceFmt,
        *bufferPtr
    );
    if(ret != FAST_OK)
        return InitFailed("fastExportToDeviceCreate for viewport bitmap failed",ret);

    cudaMemoryInfo("Created hExportToDevice");

    if(srcSurfaceFmt != FAST_RGB8 && info)
        qDebug("fastExportToDeviceCreate for viewport bitmap returned invalid format = %u", srcSurfaceFmt);


    unsigned maxPitch = 3 * ( ( ( options.MaxWidth + FAST_ALIGNMENT - 1 ) / FAST_ALIGNMENT ) * FAST_ALIGNMENT );
    unsigned bufferSize = maxPitch * options.MaxHeight * sizeof(unsigned char);
    if(cudaMalloc( &hGLBuffer, bufferSize ) != cudaSuccess)
    {
        hGLBuffer = nullptr;
        return InitFailed("cudaMalloc failed",ret);
    }

    stats["totalViewportMemory"] = bufferSize;
    cudaMemoryInfo("Created hGLBuffer");

    //JPEG Stuff
    if( true )
    {
        jfifInfo.restartInterval = options.JpegRestartInterval;
        jfifInfo.jpegFmt = FAST_JPEG_Y;
        jfifInfo.jpegMode =  FAST_JPEG_SEQUENTIAL_DCT;
        ret = fastJpegEncoderCreate(
                    &hJpegEncoder,

                    maxWidth,
                    maxHeight,

                    dstGrayBuffer
                    );

        if(ret != FAST_OK)
            return InitFailed("fastJpegEncoderCreate failed", ret);

        unsigned pitch = 3 * (((maxWidth + FAST_ALIGNMENT - 1) / FAST_ALIGNMENT) * FAST_ALIGNMENT);

        try
        {
            fastMalloc(reinterpret_cast<void**>(&jfifInfo.h_Bytestream), pitch * maxHeight * sizeof(unsigned char));
            hJpegStream.reset(static_cast<unsigned char *>(alloc.allocate(pitch * maxHeight + JPEG_HEADER_SIZE)));
        }
        catch(...)
        {
            return InitFailed("Cannot allocate memory for JPEG stream",ret);
        }

        jpegStreamSize = pitch * maxHeight + JPEG_HEADER_SIZE;

    }

    size_t requestedMemSpace = 0;
    size_t tmp = 0;
    if( hDeviceToDeviceAdapter != nullptr )
    {
        fastImportFromDeviceGetAllocatedGpuMemorySize( hDeviceToDeviceAdapter, &tmp );
        requestedMemSpace += tmp;
    }

    if( hRawUnpacker != nullptr )
    {
        fastRawImportFromDeviceGetAllocatedGpuMemorySize( hRawUnpacker, &tmp );
        requestedMemSpace += tmp;
    }

    if( hBpc != nullptr )
    {
        fastImageFiltersGetAllocatedGpuMemorySize( hBpc, &tmp );
        requestedMemSpace += tmp;
    }

    if( hLinearizationLut != nullptr )
    {
        fastImageFiltersGetAllocatedGpuMemorySize( hLinearizationLut, &tmp );
        requestedMemSpace += tmp;
    }
    if( hSam != nullptr )
    {
        fastImageFiltersGetAllocatedGpuMemorySize( hSam, &tmp );
        requestedMemSpace += tmp;
    }
    if( hDenoise != nullptr )
    {
        fastDenoiseGetAllocatedGpuMemorySize( hDenoise, &tmp );
        requestedMemSpace += tmp;
    }

    if( h16to8Transform != nullptr )
    {
        fastSurfaceConverterGetAllocatedGpuMemorySize( h16to8Transform, &tmp );
        requestedMemSpace += tmp;
    }

    if( hGrayToRGBTransform != nullptr )
    {
        fastSurfaceConverterGetAllocatedGpuMemorySize( hGrayToRGBTransform, &tmp );
        requestedMemSpace += tmp;
    }

    if(hJpegEncoder != nullptr)
    {
        fastJpegEncoderGetAllocatedGpuMemorySize(hJpegEncoder, &tmp);
        requestedMemSpace += tmp;
    }

    size_t freeMem  = 0;
    size_t totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);

    stats[QStringLiteral("totalMem")] = totalMem;
    stats[QStringLiteral("freeMem")] = freeMem;
    stats[QStringLiteral("allocatedMem")] = requestedMemSpace;

    emit initialized(QString());
    mInitialised = true;

    mut.unlock();

    return FAST_OK;
}

fastStatus_t CUDAProcessorGray::Transform(GPUImage_t *image, CUDAProcessorOptions &opts)
{
    QMutexLocker locker(&mut);

    if(image == nullptr)
    {
        mLastError = FAST_INVALID_VALUE;
        mErrString = QStringLiteral("Got null pointer data");
        return mLastError;
    }

    float fullTime = 0.;
    float elapsedTimeGpu = 0.;

    QElapsedTimer cpuTimer;
    cpuTimer.start();

    if(!mInitialised)
        return mLastError;

    mErrString = QString();
    mLastError = FAST_OK;
    fastGpuTimerHandle_t profileTimer = nullptr;
    if(info)
        fastGpuTimerCreate(&profileTimer);

    stats[QStringLiteral("hHostToDeviceAdapter")] = -1;
    stats[QStringLiteral("hRawUnpacker")] = -1;
    stats[QStringLiteral("hSam")] = -1;
    stats[QStringLiteral("hBpc")] = -1;
    stats[QStringLiteral("hWhiteBalance")] = -1;
    stats[QStringLiteral("hDebayer")] = -1;
    stats[QStringLiteral("hDenoise")] = -1;

    stats[QStringLiteral("hOutLut")] = -1;
    stats[QStringLiteral("h16to8Transform")] = -1;
    stats[QStringLiteral("hCrop")] = -1;
    stats[QStringLiteral("hDeviceToHostAdapter")] = -1;
    stats[QStringLiteral("hExportToDevice")] = -1;
    stats[QStringLiteral("hMjpegEncoder")] = -1;
    stats[QStringLiteral("totalTime")] = -1;
    stats[QStringLiteral("totalFps")] = -1;
    stats[QStringLiteral("totalGPUTime")] = -1;
    stats[QStringLiteral("totalGPUCPUTime")] = -1;

    fastStatus_t ret = FAST_OK;
    unsigned imgWidth  = image->w;
    unsigned imgHeight = image->h;

    if(imgWidth > opts.MaxWidth || imgHeight > opts.MaxHeight )
        return TransformFailed("Unsupported image size",FAST_INVALID_FORMAT,profileTimer);

    stats[QStringLiteral("inputWidth")] = imgWidth;
    stats[QStringLiteral("inputHeight")] = imgHeight;

    if(info)
        fastGpuTimerStart(profileTimer);
    QString key;
    if(hDeviceToDeviceAdapter != nullptr)
    {
        key = QStringLiteral("hHostToDeviceAdapter");
        ret = fastImportFromDeviceCopy(
                    hDeviceToDeviceAdapter,

                    image->data.get(),
                    imgWidth,
                    image->wPitch,
                    imgHeight
                    );

        if(ret != FAST_OK)
            return TransformFailed("fastImportFromHostCopy failed", ret, profileTimer);
    }
    else if(hRawUnpacker != nullptr)
    {
        key = QStringLiteral("hRawUnpacker");
        ret = fastRawImportFromDeviceDecode(
                    hRawUnpacker,

                    image->data.get(),
                    imgWidth * 3 / 2, //for packed 12 bit
                    imgWidth,
                    imgHeight
                    );

        if(ret != FAST_OK)
            return TransformFailed("fastRawUnpackerDecode failed", ret, profileTimer);
    }

    if(info)
    {
        fastGpuTimerStop(profileTimer);
        fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

        fullTime += elapsedTimeGpu;
        stats[key] = elapsedTimeGpu;
    }

    if(hSam && hSamMux)
    {
        if(opts.EnableSAM)
        {
            if(info)
            {
                fastGpuTimerStart(profileTimer);
            }

            if(image->surfaceFmt == FAST_I8)
            {
                fastSam_t samParameter;
                samParameter.correctionMatrix = nullptr;
                samParameter.blackShiftMatrix = nullptr;
                ret = fastImageFiltersTransform(
                           hSam,
                           &samParameter,
                           imgWidth,
                           imgHeight
                           );
            }
            else
            {
                fastSam16_t samParameter;
                samParameter.correctionMatrix = nullptr;
                samParameter.blackShiftMatrix = nullptr;
                ret = fastImageFiltersTransform(
                           hSam,
                           &samParameter,
                           imgWidth,
                           imgHeight
                           );
            }
            if(ret != FAST_OK && info)
                return TransformFailed("fastImageFiltersTransform for SAM failed",ret,profileTimer);

            if(info)
            {
                fastGpuTimerStop(profileTimer);
                fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

                fullTime += elapsedTimeGpu;
                stats[QStringLiteral("hSAM")] = elapsedTimeGpu;
            }

            fastMuxSelect(hSamMux, 1);
        }
        else
        {
            stats[QStringLiteral("hSAM")] = -1;
            fastMuxSelect(hSamMux, 0);
        }
    }

    unsigned short whiteLevel = opts.WhiteLevel;
    unsigned short blackLevel = opts.BlackLevel;
    if(hLinearizationLut)
    {
        if(info)
            fastGpuTimerStart(profileTimer);

        double scale = opts.eV / (double(whiteLevel - blackLevel));
        fastSurfaceFormat_t fmt = image->surfaceFmt;
        if(fmt == FAST_I8)
        {
            fastLut_8_16_t lutParameter;
            InitLut<fastLut_8_16_t>(lutParameter, blackLevel, scale, opts.LinearizationLut);
            ret = fastImageFiltersTransform(
                        hLinearizationLut,
                        &lutParameter,

                        imgWidth, imgHeight
                        );

        }
        else if(fmt == FAST_I10)
        {
            fastLut_10_t lutParameter;
            InitLut<fastLut_10_t>(lutParameter,blackLevel,scale,opts.LinearizationLut);
            ret = fastImageFiltersTransform(
                        hLinearizationLut,
                        &lutParameter,

                        imgWidth, imgHeight
                        );
        }

        else if(fmt == FAST_I12)
        {
            fastLut_12_t lutParameter;
            InitLut<fastLut_12_t>(lutParameter,blackLevel,scale,opts.LinearizationLut);
            ret = fastImageFiltersTransform(
                        hLinearizationLut,
                        &lutParameter,

                        imgWidth, imgHeight
                        );

        }
        else if(fmt == FAST_I14)
        {
            fastLut_16_t lutParameter;
            InitLut<fastLut_16_t>(lutParameter,blackLevel,scale,opts.LinearizationLut);
            ret = fastImageFiltersTransform(
                        hLinearizationLut,
                        &lutParameter,

                        imgWidth, imgHeight
                        );
        }
        else
        {
            fastLut_16_t lutParameter;
            InitLut<fastLut_16_t>(lutParameter,blackLevel,scale,opts.LinearizationLut);
            ret = fastImageFiltersTransform(
                        hLinearizationLut,
                        &lutParameter,

                        imgWidth, imgHeight
                        );
        }
        if(ret != FAST_OK && info)
            return TransformFailed("fastImageFiltersTransform for Linearization Lut failed",ret,profileTimer);

        if(info)
        {
            fastGpuTimerStop(profileTimer);
            fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

            stats[QStringLiteral("hLinearizationLut")] = elapsedTimeGpu;
            fullTime += elapsedTimeGpu;
        }
    }

    if(hBpc && hBpcMux)
    {
        if(opts.EnableBPC)
        {
            if(info)
            {
                fastGpuTimerStart(profileTimer);
            }

            fastBadPixelCorrection_t bpcParameter;
            bpcParameter.pattern = opts.BayerFormat;

            ret = fastImageFiltersTransform(
                       hBpc,
                       &bpcParameter,
                       imgWidth,
                       imgHeight
                       );

            if(ret != FAST_OK && info)
                return TransformFailed("fastImageFiltersTransform for BPC failed", ret, profileTimer);

            if(info)
            {
                fastGpuTimerStop(profileTimer);
                fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

                fullTime += elapsedTimeGpu;
                stats[QStringLiteral("hBpc")] = elapsedTimeGpu;
            }

            fastMuxSelect(hBpcMux, 1);
        }
        else
        {
            stats[QStringLiteral("hBpc")] = -1;
            fastMuxSelect(hBpcMux, 0);
        }
    }

    //Denoise
    if(hDenoise && hDenoiseMux)
    {
        if(opts.EnableDenoise)
        {
            if(info)
            {
                fastGpuTimerStart(profileTimer);;
            }


            ret = ( fastDenoiseTransform(
                        hDenoise,
                        &opts.DenoiseParams,
                        imgWidth,
                        imgHeight
                        ) );

            if(ret != FAST_OK && info)
                return TransformFailed("fastDenoiseTransform failed",ret,profileTimer);

            fastMuxSelect(hDenoiseMux, 1);

            if(info)
            {
                fastGpuTimerStop(profileTimer);
                fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

                fullTime += elapsedTimeGpu;
                stats[QStringLiteral("hDenoise")] = elapsedTimeGpu;
            }
        }
        else
        {
            stats[QStringLiteral("hDenoise")] = -1;
            fastMuxSelect(hDenoiseMux, 0);
        }
    }

    //Output LUT (gamma)
    if(hOutLut)
    {
        if(info)
        {
            fastGpuTimerStart(profileTimer);;
        }
        ret = fastImageFiltersTransform(
                    hOutLut,
                    &outLut,

                    imgWidth,
                    imgHeight
                    );

        if(ret != FAST_OK)
            return TransformFailed("fastImageFiltersTransform for output Lut failed",ret,profileTimer);

        if(info)
        {
            fastGpuTimerStop(profileTimer);
            fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

            fullTime += elapsedTimeGpu;
            stats[QStringLiteral("hOutLut")] = elapsedTimeGpu;
        }
    }

    //16-bit to 8 bit transform
    if(h16to8Transform)
    {
        if(info)
            fastGpuTimerStart(profileTimer);

        fastBitDepthConverter_t conv;
        conv.targetBitsPerChannel = 8;
        conv.isOverrideSourceBitsPerChannel = false;
        ret = fastSurfaceConverterTransform(
                    h16to8Transform,
                    &conv,

                    imgWidth,
                    imgHeight
                    );

        if(ret != FAST_OK)
            return TransformFailed("h16to8Transform transform failed",ret,profileTimer);

        if(info)
        {
            fastGpuTimerStop(profileTimer);
            fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

            fullTime += elapsedTimeGpu;
            stats[QStringLiteral("h16to8Transform")] = elapsedTimeGpu;
        }
    }



    //8 bit gray to 8 bit RGB transform
    if(hGrayToRGBTransform)
    {
        if(info) {
            fastGpuTimerStart(profileTimer);
        }

        fastBitDepthConverter_t conv;
        conv.targetBitsPerChannel = 8;
        conv.isOverrideSourceBitsPerChannel = false;
        ret = ( fastSurfaceConverterTransform(
                    hGrayToRGBTransform,
                    &conv,

                    imgWidth,
                    imgHeight
                    ));

        if(ret != FAST_OK)
            return TransformFailed("hGrayToRGBTransform transform failed",ret,profileTimer);

        if(info)
        {
            fastGpuTimerStop(profileTimer);
            fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

            fullTime += elapsedTimeGpu;
            stats[QStringLiteral("hGrayToRGBTransform")] = elapsedTimeGpu;
        }
    }

    if(hExportToDevice)
    {
        if(info)
            fastGpuTimerStart(profileTimer);

        fastExportParameters_t p;
        p.convert = FAST_CONVERT_NONE;

        ret = fastExportToDeviceCopy(
                    hExportToDevice,

                    hGLBuffer,
                    imgWidth,
                    imgWidth * 3 * sizeof(char),
                    imgHeight,
                    &p
                    );

        if(ret != FAST_OK && info)
            qDebug("fastExportToDeviceCopy failed, ret = %d", ret);


        if(info)
        {
            fastGpuTimerStop(profileTimer);
            fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

            fullTime += elapsedTimeGpu;
            stats[QStringLiteral("hExportToDevice")] = elapsedTimeGpu;
        }
    }

    if(info)
    {
        cudaDeviceSynchronize();

        float mcs = float(cpuTimer.elapsed());
        stats[QStringLiteral("totalGPUCPUTime")] = mcs;
        stats[QStringLiteral("totalGPUTime")] = fullTime;
    }

    locker.unlock();

    if(profileTimer)
    {
        fastGpuTimerDestroy(profileTimer);
        profileTimer = nullptr;
    }

    mut2.lock();
    stats2 = stats;
    mut2.unlock();

    emit finished();

    return FAST_OK;
}

fastStatus_t CUDAProcessorGray::export8bitData(void* dstPtr, bool forceRGB)
{
    fastDeviceSurfaceBufferInfo_t bufferInfo;
    fastGetDeviceSurfaceBufferInfo(dstBuffer, &bufferInfo);

    unsigned int pitch = (((bufferInfo.width + FAST_ALIGNMENT - 1) / FAST_ALIGNMENT) * FAST_ALIGNMENT) * sizeof(unsigned char);

    if(forceRGB)
        pitch *= 3;

    fastExportParameters_t p;
    p.convert = FAST_CONVERT_NONE;
    fastStatus_t ret = FAST_OK;
    if(forceRGB)
    {
        ret = (fastExportToHostCopy(
                   hDeviceToHostAdapter,
                   dstPtr,
                   bufferInfo.width,
                   pitch,
                   bufferInfo.height,
                   &p
                   ));
    }
    else
    {
        ret = (fastExportToHostCopy(
                   hDeviceToHostGrayAdapter,
                   dstPtr,
                   bufferInfo.width,
                   pitch,
                   bufferInfo.height,
                   &p
                   ));
    }

    if(ret != FAST_OK)
    {
        mErrString = QStringLiteral("fastExportToHostCopy for 8 bit data failed");
        mLastError = ret;
        emit error();
    }

    return ret;
}

//fastStatus_t CUDAProcessorGray::exportJPEGData(void* dstPtr, unsigned jpegQuality, unsigned& size)
//{
//    stats[QStringLiteral("hMjpegEncoder")] = -1;

//    if(!mInitialised || hJpegEncoder == nullptr)
//    {
//        size = 0;
//        return FAST_INVALID_HANDLE;
//    }

//    fastStatus_t ret = FAST_OK;
//    float elapsedTimeGpu = 0.;
//    fastGpuTimerHandle_t profileTimer = nullptr;
//    if(info)
//        fastGpuTimerCreate(&profileTimer);

//    fastDeviceSurfaceBufferInfo_t bufferInfo;
//    fastGetDeviceSurfaceBufferInfo(dstGrayBuffer, &bufferInfo);

//    jfifInfo.width = bufferInfo.width;
//    jfifInfo.height = bufferInfo.height;

//    if(info)
//    {
//        fastGpuTimerStart(profileTimer);
//    }
//    ret = fastJpegEncode(
//                hJpegEncoder,

//                jpegQuality,
//                &jfifInfo
//                );
//    if(ret != FAST_OK)
//        return TransformFailed("fastJpegEncode failed", ret, profileTimer);

//    ret = fastJfifStoreToMemory(reinterpret_cast<unsigned char*>(dstPtr), &size, &jfifInfo);
//    if(ret != FAST_OK)
//        return TransformFailed("fastJpegEncode failed", ret, profileTimer);

//    if(info)
//    {
//        fastGpuTimerStop(profileTimer);
//        fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);
//        stats[QStringLiteral("hMjpegEncoder")] = elapsedTimeGpu;
//    }
//    else
//    {
//        stats[QStringLiteral("hMjpegEncoder")] = -1;
//    }

//    if(profileTimer)
//    {
//        fastGpuTimerDestroy(profileTimer);
//        profileTimer = nullptr;
//    }

//    return FAST_OK;
//}
