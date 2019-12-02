#include "CUDAProcessorGray.h"

#ifdef STATIC_BUILD
extern "C"  fastStatus_t fastEnableWatermark(bool isEnabled);
#endif



CUDAProcessorGray::CUDAProcessorGray(bool info, QObject* parent) : CUDAProcessorBase(info, parent)
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
    ret = fastExperimentalImageFilterLibraryInit(handle);
    ret = fastNppGeometryLibraryInit(handle);


    if(Globals::gEnableLog)
    {
        if(QDir().mkpath(Globals::getLogPath()))
        {
            fastTraceCreate(QDir::toNativeSeparators(
                                QStringLiteral("%1%2.log").arg(Globals::getLogPath(),QDateTime::currentDateTime().
                                                               toString(QStringLiteral("dd_MM_yyyy_hh_mm_ss")))).toStdString().c_str());
            fastTraceEnableLUTDump(false);
            fastTraceEnableFlush(true);
            fastEnableInterfaceSynchronization(true);
        }
    }

    stats[QStringLiteral("inputWidth")] = -1;
    stats[QStringLiteral("inputHeight")] = -1;
    lastWidth = 0;
    lastHeight = 0;
    lastScale = QSizeF(1., 1.);

    fastSurfaceFormat_t srcSurfaceFmt  = options.SurfaceFmt;
    fastSurfaceFormat_t fmt;

    unsigned int maxWidth = options.MaxWidth;
    unsigned int maxHeight = options.MaxHeight;

    fastDeviceSurfaceBufferHandle_t *bufferPtr = nullptr;
    fastDeviceSurfaceBufferHandle_t *bypassbufferPtr = nullptr;


    ret = ( fastImportFromHostCreate(
                &hHostToDeviceAdapter,

                srcSurfaceFmt,
                maxWidth,
                maxHeight,

                &srcBuffer
                ) );
    //
    if(ret != FAST_OK)
        return InitFailed("fastImportFromHostCreate failed",ret);
    //
    cudaMemoryInfo("Created hHostToDeviceAdapter");
    bufferPtr = &srcBuffer;



    unsigned short whiteLevel = options.whiteLevel;
    unsigned short blackLevel = options.blackLevel;
    double scale = 1. / (double(whiteLevel - blackLevel));
    if(options.SurfaceFmt == FAST_I8)
    {
        fastLut_8_16_t lutParameter;
        InitLut<fastLut_8_16_t>(lutParameter,blackLevel,scale,options.linearizationLut);
        ret = ( fastImageFilterCreate(
                    &hLinearizationLut,
                    FAST_LUT_8_16,
                    &lutParameter,

                    maxWidth,
                    maxHeight,

                    *bufferPtr,
                    &linearizationLutBuffer
                    ) );
    }
    else if(options.SurfaceFmt == FAST_I10)
    {
        fastLut_10_t lutParameter;
        InitLut<fastLut_10_t>(lutParameter,blackLevel,scale,options.linearizationLut);
        ret = ( fastImageFilterCreate(
                    &hLinearizationLut,
                    FAST_LUT_10_16,
                    &lutParameter,

                    maxWidth,
                    maxHeight,

                    *bufferPtr,
                    &linearizationLutBuffer
                    ) );
    }
    else if(options.SurfaceFmt == FAST_I12)
    {
        fastLut_12_t lutParameter;
        InitLut<fastLut_12_t>(lutParameter,blackLevel,scale,options.linearizationLut);
        ret = ( fastImageFilterCreate(
                    &hLinearizationLut,
                    FAST_LUT_12_16,
                    &lutParameter,

                    maxWidth,
                    maxHeight,

                    *bufferPtr,
                    &linearizationLutBuffer
                    ) );
    }
    else if(options.SurfaceFmt == FAST_I14)
    {
        fastLut_16_t lutParameter;
        InitLut<fastLut_16_t>(lutParameter,blackLevel,scale,options.linearizationLut);
        ret = ( fastImageFilterCreate(
                    &hLinearizationLut,
                    FAST_LUT_14_16,
                    &lutParameter,

                    maxWidth,
                    maxHeight,

                    *bufferPtr,
                    &linearizationLutBuffer
                    ) );
    }
    else
    {
        fastLut_16_t lutParameter;
        InitLut<fastLut_16_t>(lutParameter,blackLevel,scale,options.linearizationLut);
        ret = ( fastImageFilterCreate(
                    &hLinearizationLut,

                    FAST_LUT_16_16,
                    &lutParameter,

                    maxWidth,
                    maxHeight,

                    *bufferPtr,
                    &linearizationLutBuffer
                    ) );
    }
    //
    if(ret != FAST_OK)
        return InitFailed("fastImageFilterCreate for Linearization Lut failed",ret);
    //
    cudaMemoryInfo("Created hLinearizationLut");
    bufferPtr = &linearizationLutBuffer;
    bypassbufferPtr = &linearizationLutBuffer;

    srcSurfaceFmt = FAST_I16;

    if(procFlags.testFlag(psMAD))
    {
        fastSam16_t samParameter;
        samParameter.correctionMatrix = options.MatrixA;
        samParameter.blackShiftMatrix = options.MatrixB;

        ret = (fastImageFilterCreate(
                   &hSam,

                   FAST_SAM16,
                   &samParameter,

                   maxWidth,
                   maxHeight,

                   linearizationLutBuffer,
                   &samBuffer
                   ) );

        if(ret != FAST_OK)
            return InitFailed("fastImageFilterCreate for MAD failed",ret);

        cudaMemoryInfo("Created MAD");

        if(linearizationLutBuffer && samBuffer)
        {
            fastDeviceSurfaceBufferHandle_t srcBuffers[2] = {linearizationLutBuffer, samBuffer};

            ret = fastMuxCreate(
                        &hSamMux,
                        srcBuffers,
                        2,
                        &samMuxBuffer
                        );
        }

        bufferPtr = &samMuxBuffer;

        //if MAD is enabled get linearized data from MAD buffer
        linearizationLutBuffer = samMuxBuffer;

        if(ret != FAST_OK)
            return InitFailed("fastMuxCreate hMadMux failed",ret);
        cudaMemoryInfo("Created hMadMux");

    }

    //median filter
    if(procFlags.testFlag(psMedianFilter))
    {
        ret = fastImageFilterCreate(
                    &hColorMedian,

                    FAST_MEDIAN,
                    nullptr,

                    maxWidth,
                    maxHeight,

                    *bufferPtr,
                    &colorMedianBuffer
                    );

        if(ret != FAST_OK)
            return InitFailed("fastImageFilterCreate for hColorMedian failed",ret);
        cudaMemoryInfo("Created hColorMedian");
        bufferPtr = &colorMedianBuffer;

        if(hLinearizationLut && hColorMedian)
        {
            fastDeviceSurfaceBufferHandle_t srcBuffers[2] = {linearizationLutBuffer,colorMedianBuffer};
            ret = fastMuxCreate(
                        &hColorMedianMux,
                        srcBuffers,
                        2,
                        &colorMedianMuxBuffer
                        );

            bufferPtr = &colorMedianMuxBuffer;
            bypassbufferPtr = &colorMedianMuxBuffer;

            if(ret != FAST_OK)
                return InitFailed("fastMuxCreate hColorMedianMux failed",ret);
            cudaMemoryInfo("Created hColorMedianMux");
        }

    }

    //Denoise
    if(procFlags.testFlag(psDenoise))
    {
        denoise_static_parameters_t denoiseParameters;
        memcpy(&denoiseParameters, &options.denoiseStaticParams, sizeof(denoise_static_parameters_t));

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
        if(bypassbufferPtr && hDenoise)
        {
            fastDeviceSurfaceBufferHandle_t srcBuffers[2] = {*bypassbufferPtr,denoiseBuffer};

            ret = fastMuxCreate(
                        &hDenoiseMux,
                        srcBuffers,
                        2,
                        &muxBuffer
                        );

            bufferPtr = &muxBuffer;

            if(ret != FAST_OK)
                return InitFailed("fastMuxCreate failed",ret);
            cudaMemoryInfo("Created hDenoiseMux");
        }
    }

    //Output LUT
    fastLut_16_t lutGray;
    for(int i = 0; i < 16384; i++)
        lutGray.lut[i] = static_cast<unsigned short>(i * 4);//rgbLut[0][i];
    //
    ret = ( fastImageFilterCreate(
                &hRawLut,

                FAST_LUT_16_16,
                &lutGray,

                maxWidth,
                maxHeight,

                *bufferPtr,
                &rawLutBuffer
                ) );

    if(ret != FAST_OK)
        return InitFailed("fastImageFilterCreate for RawCurve failed",ret);

    cudaMemoryInfo("Created rawLut");
    bufferPtr = &rawLutBuffer;



    if(procFlags.testFlag(psRemap))
    {
        fastNPPRemap_t remapData;
        remapData.map = &(options.maps);
        remapData.map->dstWidth = maxWidth;
        remapData.map->dstHeight = maxHeight;
        remapData.background = &(options.remapBackground);
        ret = fastNppGeometryCreate(
                    &hRemap,
                    FAST_NPP_GEOMETRY_REMAP,
                    NPP_INTER_CUBIC,
                    &remapData,
                    maxWidth,
                    maxHeight,
                    *bufferPtr,
                    &remapBuffer
                    );

        if(ret != FAST_OK)
            return InitFailed("fastNppGeometryCreate failed",ret);
        cudaMemoryInfo("Created hRemap");
        bufferPtr = &remapBuffer;

        //Remap mux
        if(hRawLut && hRemap)
        {
            fastDeviceSurfaceBufferHandle_t srcBuffers[2] = {rawLutBuffer,remapBuffer};

            ret = fastMuxCreate(
                        &hRemapMux,
                        srcBuffers,
                        2,
                        &remapMuxBuffer
                        );

            bufferPtr = &remapMuxBuffer;
            if(ret != FAST_OK)
                return InitFailed("fastMuxCreate hRemapMux failed",ret);
            cudaMemoryInfo("Created hRemapMux");
        }
    }

    ret = ( fastCropCreate(
                &hCrop,

                maxWidth,
                maxHeight,

                maxWidth,
                maxHeight,

                *bufferPtr,
                &cropBuffer
                ) );
    bufferPtr = &cropBuffer;

    if(ret != FAST_OK)
        return InitFailed("fastCropCreate failed",ret);

    cudaMemoryInfo("Created hCrop");

    unsigned maxOutWidth = options.maxOutWidth <= 0 ? Globals::getAppSettings()->maxOutWidth : options.maxOutWidth;
    unsigned maxOutHeight = options.maxOutHeight <= 0 ? Globals::getAppSettings()->maxOutHeight : options.maxOutHeight;
    bool maxSizeChanged = maxOutWidth < maxWidth || maxOutHeight < maxHeight;
    if(maxSizeChanged)
    {
        if(maxOutWidth < maxWidth)
            maxOutWidth = maxWidth;
        if(maxOutHeight < maxHeight)
            maxOutHeight = maxHeight;
    }


    ret = fastResizerCreate(
                &hResizer,

                maxWidth,
                maxHeight,

                maxOutWidth,
                maxOutHeight,

                10,

                0.5,
                0.5,

                *bufferPtr,
                &resizerBuffer
                );

    if(ret != FAST_OK)
        return InitFailed("fastResizerCreate failed",ret);

    cudaMemoryInfo("Created hResizer");

    maxWidth = maxOutWidth;
    maxHeight = maxOutHeight;
    bufferPtr = &resizerBuffer;

    //USM
    if(procFlags.testFlag(psUSM))
    {
        ret = fastNppFilterCreate(
                    &hUSM,

                    NPP_UNSHARP_MASK_SOFT,
                    nullptr,

                    maxWidth,
                    maxHeight,

                    *bufferPtr,
                    &usmBuffer
                    );
        bufferPtr = &usmBuffer;

        if(ret != FAST_OK && info)
            return InitFailed("fastImageFilterCreate for USM failed",ret);

        cudaMemoryInfo("Created hUSM");

        //USM mux
        if(hUSM && hResizer)
        {
            fastDeviceSurfaceBufferHandle_t srcBuffers[2] = {resizerBuffer,usmBuffer};

            ret = fastMuxCreate(
                        &hUSMMux,
                        srcBuffers,
                        2,
                        &usmMuxBuffer
                        );

            bufferPtr = &usmMuxBuffer;

            if(ret != FAST_OK && info)
                return InitFailed("fastMuxCreate hUSMMux failed",ret);

            cudaMemoryInfo("Created hUSMMux");
        }
    }

    //16 bit RGB untransformed data export
    ret = ( fastExportToHostCreate(
                &hDeviceToHost16NoTransformAdapter,
                &srcSurfaceFmt,
                *bufferPtr
                ) );

    if(ret != FAST_OK)
        return InitFailed("fastExportToHostCreate for 16 bit data failed",ret);

    cudaMemoryInfo("Created hDeviceToHost16NoTransformAdapter");

    if(srcSurfaceFmt != FAST_RGB16 && info)
        qDebug("hDeviceToHost16NoTransformAdapter returned invalid format = %u", srcSurfaceFmt);


    ret = fastAffineCreate(
                &hAffineTransform,

                FAST_AFFINE_ALL,

                maxWidth,
                maxHeight,

                *bufferPtr,
                &affineTransformBuffer
                );
    bufferPtr = &affineTransformBuffer;

    if(ret != FAST_OK && info)
        return InitFailed("fastAffineCreate failed",ret);

    maxWidth = maxHeight = qMax(maxWidth, maxHeight);
    cudaMemoryInfo("Created hAffineTransform");

    //16 bit affine transformed RGB data export
    ret = ( fastExportToHostCreate(
                &hDeviceToHost16Adapter,
                &srcSurfaceFmt,
                *bufferPtr
                ) );

    if(ret != FAST_OK)
        return InitFailed("fastExportToHostCreate for 16 bit data failed",ret);

    cudaMemoryInfo("Created hDeviceToHost16Adapter");

    if(srcSurfaceFmt != FAST_RGB16 && info)
        qDebug("hDeviceToHost16Adapter returned invalid format = %u", srcSurfaceFmt);

    try
    {
        rgb16Bits.reset(static_cast<unsigned char*>(alloc.allocate(options.MaxWidth * options.MaxHeight * 3 * sizeof(unsigned short))));
    }
    catch(...)
    {
        return InitFailed("Cannot allocate memory for RGB 16 bitmap",ret);
    }

    //Linearized raw data export
    ret = ( fastExportToHostCreate(
                &hDeviceToHostLinRawAdapter,
                &srcSurfaceFmt,
                linearizationLutBuffer
                ) );

    if(ret != FAST_OK)
        return InitFailed("fastExportToHostCreate for RAW linearized data failed",ret);

    cudaMemoryInfo("Created hDeviceToHostLinRawAdapter");

    if(srcSurfaceFmt != FAST_I16 && info)
        qDebug("hDeviceToHostLinRawAdapter returned invalid format = %u", srcSurfaceFmt);


    //16 to 8 bit transform
    fastBitDepthConverter_t conv;
    conv.bitsPerChannel = 8;
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

    if(srcSurfaceFmt != FAST_RGB8 && info)
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


    fastBayerPatternParam_t histParams;

    histParams.bayerPattern = FAST_BAYER_NONE;
    ret = ( fastHistogramCreate(
                &hHistogram,
                FAST_HISTOGRAM_COMMON,
                &histParams,
                HIST_BINS,
                maxWidth,
                maxHeight,

                dstGrayBuffer
                ) );

    try
    {
        histResults.reset(static_cast<unsigned int *>(alloc.allocate(3 * HIST_BINS * sizeof(unsigned int))));
    }
    catch(...)
    {
        return InitFailed("Cannot allocate memory for histogram",ret);
    }

    if(ret != FAST_OK)
        return InitFailed("fastHistogramCreate for histogram failed",ret);

    cudaMemoryInfo("Created hHistogram");

    fastHistogramParade_t paradeParams;
    paradeParams.stride = paradeStride;

    ret = ( fastHistogramCreate(
                &hRGBParade,
                FAST_HISTOGRAM_PARADE,
                &paradeParams,
                HIST_BINS,
                maxWidth,
                maxHeight,

                *bufferPtr
                ) );

    try
    {
        paradeResults.reset(static_cast<unsigned int *>(alloc.allocate(maxWidth * 3 * HIST_BINS * sizeof(unsigned int))));
    }
    catch(...)
    {
        return InitFailed("Cannot allocate memory for parade",ret);
    }

    if(ret != FAST_OK)
    {
        qDebug("fastHistogramsCreate hRGBParade failed, ret = %u", ret);
        return InitFailed("fastHistogramsCreate for RGB Parade failed",ret);
    }

    cudaMemoryInfo("Created hRGBParade");

    //OpenGL viewport bitmap
    ret = fastCropCreate(
                &hViewportCrop,

                maxWidth,
                maxHeight,

                maxWidth,
                maxHeight,

                *bufferPtr,
                &viewprtCropBuffer
                );

    if(ret != FAST_OK)
        return InitFailed("fastCropCreate for viewport bitmap failed",ret);

    cudaMemoryInfo("Created hViewportCrop");

    ret = fastResizerCreate(
                &hViewportResizer,

                maxWidth,
                maxHeight,

                static_cast<unsigned int>(maxViewportSize.width()),
                static_cast<unsigned int>(maxViewportSize.height()),

                MAX_ZOOM,

                0.5,
                0.5,

                viewprtCropBuffer,
                &viewportResizerBuffer
                );

    if(ret != FAST_OK)
        return InitFailed("fastResizerCreate for viewport bitmap failed",ret);

    cudaMemoryInfo("Created hViewportResizer");

    bufferPtr = &viewportResizerBuffer;

    fmt = FAST_RGB8;
    ret = ( fastExportToHostCreate(
                &hBitmapExport,
                &fmt,
                *bufferPtr
                ) );

    if(ret != FAST_OK)
        return InitFailed("fastExportToDeviceCreate for viewport bitmap failed",ret);

    cudaMemoryInfo("Created hBitmapExport");

    if(srcSurfaceFmt != FAST_I8 && info)
        qDebug("fastExportToDeviceCreate for viewport bitmap returned invalid format = %u", srcSurfaceFmt);

    if(procFlags.testFlag(psJPEG))
    {
        if(options.codec == CUDAProcessorOptions::vcMJPG)
        {
            //Check if videoFileName is not empty and its parent dir exists
            if(!options.videoFileName.isEmpty())
            {
                if(QFileInfo(options.videoFileName).dir().exists())
                {
                    jfifInfo.restartInterval = options.jpegRestartInterval;
                    jfifInfo.jpegFmt = options.jpegSamplingFmt;
                    jfifInfo.jpegMode =  JPEG_SEQUENTIAL_DCT;
                    ret = fastJpegEncoderCreate(
                                &hMjpegEncoder,

                                maxWidth,
                                maxHeight,

                                dstBuffer
                                );

                    if(ret != FAST_OK)
                        return InitFailed("fastJpegEncoderCreate failed",ret);

                    fastMJpegFileDescriptor_t fileDescriptor;

                    QString str = QDir::toNativeSeparators(options.videoFileName);
                    char filename[256];
                    memset(filename, 0, 256);
                    memcpy(filename, str.toStdString().c_str(), size_t(str.size()));

                    fileDescriptor.fileName = filename;
                    fileDescriptor.height = options.Height;
                    fileDescriptor.width = options.Width;
                    fileDescriptor.samplingFmt = options.jpegSamplingFmt;
                    ret = fastMJpegWriterCreate(
                                &hMjpegWriter,

                                &fileDescriptor,
                                int(options.videoFrameRate)
                                );

                    if(ret != FAST_OK)
                        return InitFailed("fastMJpegWriterCreate failed",ret);

                    unsigned pitch = 3 * ( ( ( maxWidth + FAST_ALIGNMENT - 1 ) / FAST_ALIGNMENT ) * FAST_ALIGNMENT );
                    fastMalloc(reinterpret_cast<void**>(&jfifInfo.h_Bytestream), pitch * maxHeight * sizeof(unsigned char));

                    try
                    {
                        hJpegStream.reset(static_cast<unsigned char *>(alloc.allocate(pitch * maxHeight + JPEG_HEADER_SIZE)));
                    }
                    catch(...)
                    {
                        return InitFailed("Cannot allocate memory for JPEG stream",ret);
                    }

                    jpegStreamSize = pitch * maxHeight + JPEG_HEADER_SIZE;
                }
            }
        }

        if(options.codec == CUDAProcessorOptions::vcJPG)
        {

            jfifInfo.restartInterval = options.jpegRestartInterval;
            jfifInfo.jpegFmt = JPEG_Y;//options.jpegSamplingFmt;
            jfifInfo.jpegMode =  JPEG_SEQUENTIAL_DCT;
            ret = fastJpegEncoderCreate(
                        &hMjpegEncoder,

                        maxWidth,
                        maxHeight,

                        dstGrayBuffer
                        );

            if(ret != FAST_OK)
                return InitFailed("fastJpegEncoderCreate failed",ret);

            unsigned pitch = 3 * ( ( ( maxWidth + FAST_ALIGNMENT - 1 ) / FAST_ALIGNMENT ) * FAST_ALIGNMENT );

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
    }

    if(maxSizeChanged)
    {
        emit outputSizeChanged(QSize(int(maxOutWidth), int(maxOutHeight)));
    }

    size_t  requestedMemSpace = 0;
    unsigned tmp = 0;
    if( hRawUnpacker != nullptr )
    {
        fastRawUnpackerGetAllocatedGpuMemorySize( hRawUnpacker, &tmp );
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
    //    if( hTempDenoiser != NULL )
    //    {
    //        fastExperimentalImageFiltersGetAllocatedGpuMemorySize( hTempDenoiser, &tmp );
    //        requestedMemSpace += tmp;
    //    }
    if( hRgbLut != nullptr )
    {
        fastImageFiltersGetAllocatedGpuMemorySize( hRgbLut, &tmp );
        requestedMemSpace += tmp;
    }
    if( h16to8Transform != nullptr )
    {
        fastSurfaceConverterGetAllocatedGpuMemorySize( h16to8Transform, &tmp );
        requestedMemSpace += tmp;
    }
    if( hHostToDeviceAdapter != nullptr )
    {
        fastImportFromHostGetAllocatedGpuMemorySize( hHostToDeviceAdapter, &tmp );
        requestedMemSpace += tmp;
    }

    if(hCrop != nullptr )
    {
        fastCropGetAllocatedGpuMemorySize( hCrop, &tmp );
        requestedMemSpace += tmp;
    }

    if(hHistogram != nullptr )
    {
        fastHistogramGetAllocatedGpuMemorySize( hHistogram, &tmp );
        requestedMemSpace += tmp;
    }

    if(hRGBParade != nullptr )
    {
        fastHistogramGetAllocatedGpuMemorySize( hRGBParade, &tmp );
        requestedMemSpace += tmp;
    }

    if(hMjpegEncoder != nullptr)
    {
        fastJpegEncoderGetAllocatedGpuMemorySize( hMjpegEncoder, &tmp );
        requestedMemSpace += tmp;
    }

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);

    stats[QStringLiteral("totalMem")] = totalMem;
    stats[QStringLiteral("freeMem")] = freeMem;
    stats[QStringLiteral("allocatedMem")] = requestedMemSpace;

    emit initialized(QString());
    mInitialised = true;

    mut.unlock();

    return FAST_OK;
}


fastStatus_t CUDAProcessorGray::Transform(Image<unsigned char, FastAllocator> &image, void *dstPtr, CUDAProcessorOptions &opts)
{
    Q_UNUSED(dstPtr)
    QMutexLocker locker(&mut);

    float fullTime = 0.;
    float elapsedTimeGpu = 0.;

    LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
    QueryPerformanceCounter(&StartingTime);

    if(!mInitialised)
        return mLastError;

    mErrString = QString();
    mLastError = FAST_OK;
    fastGpuTimerHandle_t profileTimer = nullptr;
    if(info)
        fastGpuTimerCreate(&profileTimer);


    stats[QStringLiteral("hHostToDeviceAdapter")] = -1;
    stats[QStringLiteral("hWhiteBalance")] = -1;
    stats[QStringLiteral("hRawLut")] = -1;
    stats[QStringLiteral("hDebayer")] = -1;
    stats[QStringLiteral("hDebayerCPU")] = -1;
    stats[QStringLiteral("hDenoise")] = -1;
    stats[QStringLiteral("hTempDenoiser")] = -1;
    stats[QStringLiteral("hToneCurve")] = -1;
    stats[QStringLiteral("hLookTable")] = -1;
    stats[QStringLiteral("hHueSatMap")] = -1;
    stats[QStringLiteral("hWideCSConverter")] = -1;
    stats[QStringLiteral("hOutCSConverter")] = -1;
    stats[QStringLiteral("hRgbLut")] = -1;
    stats[QStringLiteral("hResizer")] = -1;
    stats[QStringLiteral("hUSM")] = -1;
    stats[QStringLiteral("hCubeLut")] = -1;
    stats[QStringLiteral("hAffineTransform")] = -1;
    stats[QStringLiteral("hLinearizationLut")] = -1;
    stats[QStringLiteral("hGrayToRGBTransform")] = -1;

    stats[QStringLiteral("hBayerDenoise")] = -1;
    stats[QStringLiteral("hBayerSplitter")] = -1;
    stats[QStringLiteral("hRemap")] = -1;

    stats[QStringLiteral("hColorMedian")] = -1;

    stats[QStringLiteral("hHSVLut")] = -1;
    stats[QStringLiteral("hHistogram")] = -1;
    stats[QStringLiteral("hRGBParade")] = -1;

    stats[QStringLiteral("h16to8Transform")] = -1;
    stats[QStringLiteral("hCrop")] = -1;
    stats[QStringLiteral("hMjpegEncoder")] = -1;
    stats[QStringLiteral("hDeviceToHostAdapter")] = -1;
    stats[QStringLiteral("totalTime")] = -1;
    stats[QStringLiteral("totalFps")] = -1;
    stats[QStringLiteral("totalGPUTime")] = -1;
    stats[QStringLiteral("totalGPUCPUTime")] = -1;

    fastStatus_t ret;

    if( image.w > opts.MaxWidth || image.h > opts.MaxHeight )
        return TransformFailed("Unsupported image size",FAST_INVALID_FORMAT,profileTimer);

    stats[QStringLiteral("inputWidth")] = image.w;
    stats[QStringLiteral("inputHeight")] = image.h;
    lastWidth = image.w;
    lastHeight = image.h;
    lastScale = QSizeF(qreal(opts.scaleX), qreal(opts.scaleY));

    if(info) {
        fastGpuTimerStart(profileTimer);
    }

    ret = ( fastImportFromHostCopy(
                hHostToDeviceAdapter,

                image.data.get(),
                image.w,
                image.wPitch,
                image.h
                ) );

    SYNC_PIPELINE("hHostToDeviceAdapter");
    if(ret != FAST_OK)
        return TransformFailed("fastImportFromHostCopy failed",ret,profileTimer);

    if(info)
    {
        fastGpuTimerStop(profileTimer);
        fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

        fullTime += elapsedTimeGpu;
        stats[QStringLiteral("hHostToDeviceAdapter")] = elapsedTimeGpu;


    }

    unsigned int imageWidth  = image.w;
    unsigned int imageHeight = image.h;
    unsigned int cropedWidth = lastWidth = image.w;
    unsigned int cropedHeight = lastHeight = image.h;

    unsigned short whiteLevel = opts.whiteLevel;
    unsigned short blackLevel = opts.blackLevel;

    if(hLinearizationLut)
    {
        if(info)
            fastGpuTimerStart(profileTimer);
        double scale = double(opts.eV) / (double(whiteLevel - blackLevel));
        if(image.surfaceFmt == FAST_I8)
        {
            fastLut_8_16_t lutParameter;
            InitLut<fastLut_8_16_t>(lutParameter,blackLevel,scale,opts.linearizationLut);
            ret = ( fastImageFiltersTransform(
                        hLinearizationLut,
                        &lutParameter,

                        cropedWidth, cropedHeight
                        ) );

        }
        else if(image.surfaceFmt == FAST_I10)
        {
            fastLut_10_t lutParameter;
            InitLut<fastLut_10_t>(lutParameter,blackLevel,scale,opts.linearizationLut);
            ret = ( fastImageFiltersTransform(
                        hLinearizationLut,
                        &lutParameter,

                        cropedWidth, cropedHeight
                        ) );
        }

        else if(image.surfaceFmt == FAST_I12)
        {
            fastLut_12_t lutParameter;
            InitLut<fastLut_12_t>(lutParameter,blackLevel,scale,opts.linearizationLut);
            ret = ( fastImageFiltersTransform(
                        hLinearizationLut,
                        &lutParameter,

                        cropedWidth, cropedHeight
                        ) );
        }
        else if(image.surfaceFmt == FAST_I14)
        {
            fastLut_16_t lutParameter;
            InitLut<fastLut_16_t>(lutParameter,blackLevel,scale,opts.linearizationLut);
            ret = ( fastImageFiltersTransform(
                        hLinearizationLut,
                        &lutParameter,

                        cropedWidth, cropedHeight
                        ) );
        }
        else
        {
            fastLut_16_t lutParameter;
            InitLut<fastLut_16_t>(lutParameter,blackLevel,scale,opts.linearizationLut);
            ret = ( fastImageFiltersTransform(
                        hLinearizationLut,
                        &lutParameter,

                        cropedWidth, cropedHeight
                        ) );
        }
        SYNC_PIPELINE("hLinearizationLut");
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

    if(hSam && hSamMux)
    {
        if(opts.enableMad)
        {
            if(info)
            {
                fastGpuTimerStart(profileTimer);
            }

            if(opts.SurfaceFmt == FAST_I8)
            {
                fastSam_t madParameter;
                madParameter.correctionMatrix = opts.MatrixA;
                madParameter.blackShiftMatrix = (char*)opts.MatrixB;
                ret = (fastImageFiltersTransform(
                           hSam,
                           haveNewMad ? &madParameter : nullptr,
                           cropedWidth,
                           cropedHeight
                           ));
            }
            else
            {
                fastSam16_t madParameter;
                madParameter.correctionMatrix = opts.MatrixA;
                madParameter.blackShiftMatrix = opts.MatrixB;
                ret = (fastImageFiltersTransform(
                           hSam,
                           haveNewMad ? &madParameter : nullptr,
                           cropedWidth,
                           cropedHeight
                           ));
            }

            SYNC_PIPELINE("hMAD");
            if(ret != FAST_OK && info)
                return TransformFailed("fastImageFiltersTransform for MAD failed",ret,profileTimer);

            haveNewMad = false;

            if(info)
            {
                fastGpuTimerStop(profileTimer);
                fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

                fullTime += elapsedTimeGpu;
                stats[QStringLiteral("hMAD")] = elapsedTimeGpu;
            }

            fastMuxSelect(hSamMux, 1);
            SYNC_PIPELINE("hMadMux");
        }
        else
        {
            stats[QStringLiteral("hMAD")] = -1;
            fastMuxSelect(hSamMux, 0);
            SYNC_PIPELINE("hMadMux");
        }
    }

    //Color median filter
    if(hColorMedian && hColorMedianMux)
    {
        if(opts.enableColorMedian)
        {
            if(info)
            {
                fastGpuTimerStart(profileTimer);;
            }

            ret = ( fastImageFiltersTransform(
                        hColorMedian,
                        nullptr,

                        cropedWidth,
                        cropedHeight
                        ) );
            SYNC_PIPELINE("hColorMedian");

            fastMuxSelect(hColorMedianMux, 1);
            SYNC_PIPELINE("hColorMedianMux");

            if(info)
            {
                fastGpuTimerStop(profileTimer);
                fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

                fullTime += elapsedTimeGpu;
                stats[QStringLiteral("hColorMedian")] = elapsedTimeGpu;
            }

            if(ret != FAST_OK && info)
                return TransformFailed("fastImageFiltersTransform hColorMedian failed",ret,profileTimer);
        }
        else
        {
            stats[QStringLiteral("hColorMedian")] = -1;
            fastMuxSelect(hColorMedianMux, 0);
            SYNC_PIPELINE("hColorMedianMux");
        }
    }

    //Denoise
    if(hDenoise && hDenoiseMux)
    {
        if(opts.enableDenoise)
        {
            if(info)
            {
                fastGpuTimerStart(profileTimer);;
            }


            ret = ( fastDenoiseTransform(
                        hDenoise,
                        &opts.denoiseParams,
                        cropedWidth,
                        cropedHeight
                        ) );
            SYNC_PIPELINE("hDenoise");
            if(ret != FAST_OK && info)
                return TransformFailed("fastDenoiseTransform failed",ret,profileTimer);

            fastMuxSelect(hDenoiseMux, 1);
            SYNC_PIPELINE("hDenoiseMux");

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
            SYNC_PIPELINE("hDenoiseMux");
        }
    }

    //RGB LUT
    if(hRawLut)
    {
        if(info)
        {
            fastGpuTimerStart(profileTimer);;
        }
        fastLut_16_t lut;
        copy(rgbLut[0].begin(),rgbLut[0].end(),begin(lut.lut));
        ret = ( fastImageFiltersTransform(
                    hRawLut,
                    &lut,

                    cropedWidth,
                    cropedHeight
                    ) );
        SYNC_PIPELINE("hRgbLut");

        if(ret != FAST_OK)
            return TransformFailed("fastImageFiltersTransform for Rgb Lut failed",ret,profileTimer);

        if(info)
        {
            fastGpuTimerStop(profileTimer);
            fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

            fullTime += elapsedTimeGpu;
            stats[QStringLiteral("hRgbLut")] = elapsedTimeGpu;
        }
    }

    if(hRemap)
    {
        if(opts.enableRemap)
        {
            if(info)
            {
                fastGpuTimerStart(profileTimer);;
            }
            fastNPPRemap_t remapData;
            remapData.map = &(opts.maps);
            remapData.background = &(opts.remapBackground);
            remapData.background->isEnabled = true;
            remapData.map->dstWidth = cropedWidth;
            remapData.map->dstHeight = cropedHeight;
            ret = fastNppGeometryTransform(
                        hRemap,
                        &remapData,
                        cropedWidth,
                        cropedHeight
                        );
            SYNC_PIPELINE("hRemap");

            if(ret != FAST_OK)
                return TransformFailed("fastNppGeometryTransform failed",ret,profileTimer);

            if(info)
            {
                fastGpuTimerStop(profileTimer);
                fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

                fullTime += elapsedTimeGpu;
                stats[QStringLiteral("hRemap")] = elapsedTimeGpu;
            }
            fastMuxSelect(hRemapMux, 1);
            SYNC_PIPELINE("hRemapMux");

        }
        else
        {
            stats[QStringLiteral("hRemap")] = -1;
            fastMuxSelect(hRemapMux, 0);
            SYNC_PIPELINE("hRemapMux");
        }
    }

    if(hCrop)
    {
        if(info)
        {
            fastGpuTimerStart(profileTimer);;
        }
        unsigned int x = 0;
        unsigned int y = 0;
        unsigned int w = 0;
        unsigned int h = 0;
        if(opts.cropRect.isValid())
        {
            x = unsigned(opts.cropRect.left());
            y = unsigned(opts.cropRect.top());
            w = unsigned(opts.cropRect.width());
            h = unsigned(opts.cropRect.height());
            imageWidth  = unsigned(opts.cropRect.width());
            imageHeight = unsigned(opts.cropRect.height());
        }
        else
        {
            x = 0;
            y = 0;
            w = image.w;
            h = image.h;
        }

        ret = fastCropTransform(
                    hCrop,
                    image.w,
                    image.h,
                    x,
                    y,
                    w,
                    h
                    );
        SYNC_PIPELINE("hCrop");
        if(ret != FAST_OK)
            return TransformFailed("fastCropTransform failed",ret,profileTimer);

        if(info)
        {
            fastGpuTimerStop(profileTimer);
            fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

            fullTime += elapsedTimeGpu;
            stats[QStringLiteral("hCrop")] = elapsedTimeGpu;
        }
    }
    else
    {
        stats[QStringLiteral("hCrop")] = -1;
    }

    cropedWidth = lastWidth = imageWidth;
    cropedHeight = lastHeight = imageHeight;

    if(hResizer)
    {
        if(info)
        {
            fastGpuTimerStart(profileTimer);;
        }
        auto resizedWidth = unsigned(cropedWidth * opts.scaleX);
        auto resizedHeight = unsigned(cropedHeight * opts.scaleY);
        ret = fastResizerTransformStretch(
                    hResizer,
                    FAST_LANCZOS,

                    cropedWidth,
                    cropedHeight,

                    resizedWidth,
                    resizedHeight
                    );
        SYNC_PIPELINE("fastResizerTransform");

        if( ret != FAST_OK )
        {
            qDebug("Resizing image failed width = %u, height = %u, ret = %u", resizedWidth, resizedHeight, ret);
            return TransformFailed("Resizing image failed",ret,profileTimer);
        }

        if(info)
        {
            fastGpuTimerStop(profileTimer);
            fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

            stats[QStringLiteral("hResizer")] = elapsedTimeGpu;
            fullTime += elapsedTimeGpu;
        }

        cropedWidth = lastWidth = resizedWidth;
        cropedHeight = lastHeight = resizedHeight;
    }


    //USM
    if(hUSM && hUSMMux)
    {
        if(opts.enableUSM)
        {
            if(info)
            {
                fastGpuTimerStart(profileTimer);;
            }
            fastNPPUnsharpMaskFilter_t filterParameters;
            filterParameters.sigma = opts.USMSigma;
            filterParameters.amount = opts.USMValue;
            filterParameters.envelopMedian = 0.5;
            filterParameters.envelopSigma = 5;
            filterParameters.envelopRank = 4;
            filterParameters.envelopCoef = -2;
            ret = fastNppFilterTransform(
                        hUSM,
                        cropedWidth,
                        cropedHeight,

                        &filterParameters
                        );
            SYNC_PIPELINE("hUSM");

            if(ret != FAST_OK)
                return TransformFailed("fastImageFiltersTransform for USM failed",ret,profileTimer);

            fastMuxSelect(hUSMMux, 1);
            SYNC_PIPELINE("hUSMMux");

            if(info)
            {
                fastGpuTimerStop(profileTimer);
                fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

                fullTime += elapsedTimeGpu;
                stats[QStringLiteral("hUSM")] = elapsedTimeGpu;
            }
        }
        else
        {
            stats[QStringLiteral("hUSM")] = -1;
            fastMuxSelect(hUSMMux, 0);
            SYNC_PIPELINE("hUSMMux");
        }
    }
    if(hAffineTransform)
    {
        if(info)
        {
            fastGpuTimerStart(profileTimer);;
        }
        int affineType = Globals::getAffineType(opts.angle, opts.horFlip, opts.verFlip);
        if(affineType >= 0)
        {
            ret = fastAffineTransform(
                        hAffineTransform,
                        fastAffineTransformations_t(affineType),
                        cropedWidth,
                        cropedHeight
                        );
            SYNC_PIPELINE("hAffineTransform");

            if(ret != FAST_OK)
                return TransformFailed("fastAffineTransform failed",ret,profileTimer);

            if(info)
            {
                fastGpuTimerStop(profileTimer);
                fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

                fullTime += elapsedTimeGpu;
                stats[QStringLiteral("hAffineTransform")] = elapsedTimeGpu;
            }
        }
    }

    unsigned int w = 0;
    unsigned int h = 0;
    unsigned int pitch = 0;
    export16bitData(nullptr, w, h, pitch);
    cropedWidth = w;
    cropedHeight = h;

    //16 to 8 bit transform
    if(h16to8Transform)
    {
        if(info) {
            fastGpuTimerStart(profileTimer);
        }

        fastBitDepthConverter_t conv;
        conv.bitsPerChannel = 8;
        ret = ( fastSurfaceConverterTransform(
                    h16to8Transform,
                    &conv,

                    cropedWidth,
                    cropedHeight
                    ) );

        SYNC_PIPELINE("h16to8Transform");

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
        conv.bitsPerChannel = 8;
        ret = ( fastSurfaceConverterTransform(
                    hGrayToRGBTransform,
                    &conv,

                    cropedWidth,
                    cropedHeight
                    ) );

        SYNC_PIPELINE("hGrayToRGBTransform");

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

    if(hHistogram)
    {
        if(info)
        {
            fastGpuTimerStart(profileTimer);
        }

        fastBayerPatternParam_t histParams;

        histParams.bayerPattern = FAST_BAYER_NONE;
        ret = fastHistogramCalculate(hHistogram,
                                     &histParams,
                                     0,
                                     0,
                                     cropedWidth,
                                     cropedHeight,

                                     histResults.get());
        SYNC_PIPELINE("hHistogram");

        if(ret != FAST_OK)
            return TransformFailed("fastHistogramsCalculate for histogram failed",ret,profileTimer);

        if(info)
        {
            fastGpuTimerStop(profileTimer);
            fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);

            fullTime += elapsedTimeGpu;
            stats[QStringLiteral("hHistogram")] = elapsedTimeGpu;
        }
    }

    calcRGBParade(cropedWidth, cropedHeight);
    elapsedTimeGpu = stats[QStringLiteral("hRGBParade")];
    if(elapsedTimeGpu > 0)
        fullTime += elapsedTimeGpu;


    //Encode into Motion Jpeg or JPEG (if any)
    if(hMjpegEncoder)
    {
        jfifInfo.width = imageWidth;
        jfifInfo.height = imageHeight;
        unsigned jpegSize = jpegStreamSize;
        if(info)
        {
            fastGpuTimerStart(profileTimer);
        }
        ret = fastJpegEncode(
                    hMjpegEncoder,

                    opts.jpegQuality,
                    &jfifInfo
                    );
        if(ret != FAST_OK)
        {
            //qDebug("fastJpegEncode failed, ret = %u", ret);
            return TransformFailed("fastJpegEncode failed",ret,profileTimer);
        }

        //write to MJPEG
        if(hMjpegWriter)
        {
            ret = fastJfifStoreToMemory(
                        hJpegStream.get(),
                        &jpegSize,

                        &jfifInfo
                        );

            if(ret != FAST_OK)
            {
                //qDebug("fastJfifStoreToMemory failed, ret = %u", ret);
                return TransformFailed("fastJfifStoreToMemory failed",ret,profileTimer);
            }

            ret = fastMJpegWriteFrame(
                        hMjpegWriter,

                        hJpegStream.get(),
                        int(jpegSize)
                        );
            if(ret != FAST_OK)
            {
                //qDebug("fastMJpegWriteFrame failed, ret = %u", ret);
                return TransformFailed("fastMJpegWriteFrame failed",ret,profileTimer);
            }

            if( info )
            {
                fastGpuTimerStop(profileTimer);
                fastGpuTimerGetTime(profileTimer, &elapsedTimeGpu);
                stats[QStringLiteral("hMjpegEncoder")] = elapsedTimeGpu;
                fullTime += elapsedTimeGpu;
            }
        }
        //Write to Jpeg file
        else
        {
            QString fileName = QString::fromStdString(image.inputFileName);
            QString path(QDir::toNativeSeparators(QDir(QString::fromStdString(image.outputFileName)).path()));

            if(path.endsWith(QDir::separator()))
                fileName = QStringLiteral("%1%2.jpg").arg(path,QFileInfo(fileName).baseName());
            else
                fileName = QStringLiteral("%1%2%3.jpg").arg(path).arg(QDir::separator()).arg(QFileInfo(fileName).baseName());

            ret = fastJfifStoreToFile(
                        fileName.toStdString().c_str(),
                        &jfifInfo
                        );
            if(ret != FAST_OK)
                return TransformFailed("fastJfifStoreToFile failed",ret,profileTimer);
        }

    }
    else
    {
        stats[QStringLiteral("hMjpegEncoder")] = -1;
    }

    if(info)
    {
        cudaDeviceSynchronize();
        QueryPerformanceCounter(&EndingTime);
        ElapsedMicroseconds = Globals::getMcs(StartingTime, EndingTime);
        float mcs = ElapsedMicroseconds.QuadPart / 1000.f;
        stats[QStringLiteral("totalGPUCPUTime")] = mcs;
        stats[QStringLiteral("totalGPUTime")] = fullTime;
    }
    locker.unlock();

    if(profileTimer)
    {
        fastGpuTimerDestroy(profileTimer);
        profileTimer = nullptr;
    }

    emit finished(-1);
    return FAST_OK;
}

fastStatus_t CUDAProcessorGray::export8bitData(void* dstPtr, bool forceRGB)
{
    if(lastWidth <= 0 || lastHeight <= 0)
        return FAST_INVALID_SIZE;
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
        emit error(mLastError, mErrString);
    }

    return ret;
}
