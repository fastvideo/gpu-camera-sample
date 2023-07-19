#include "GeniCamCamera.h"

#ifdef SUPPORT_GENICAM
#include <QVector>
#include <QElapsedTimer>

#include <RawProcessor.h>

using CameraStatEnum = GPUCameraBase::cmrCameraStatistic  ;

GeniCamCamera::GeniCamCamera()
{
    mCameraThread.setObjectName(QStringLiteral("GeniCamThread"));
    moveToThread(&mCameraThread);
    mCameraThread.start();
}

GeniCamCamera::~GeniCamCamera()
{
    mCameraThread.quit();
    mCameraThread.wait(3000);

    if(mDevice)
        mDevice->close();
    rcg::System::clearSystems();
}

bool GeniCamCamera::open(uint32_t devID)
{
    using namespace GenApi;

    std::vector<std::shared_ptr<rcg::System> > system=rcg::System::getSystems();
    for (auto & i : system)
    {
        if(mDevice)
            break;

        i->open();
        std::vector<std::shared_ptr<rcg::Interface>> interf = i->getInterfaces();

        for (auto & k : interf)
        {
            k->open();
            std::vector<std::shared_ptr<rcg::Device>> device = k->getDevices();

            for (const auto & j : device)
            {
                mDevice = j;
                break;
            }

            if(!mDevice)
                k->close();
        }
        if(!mDevice)
            i->close();
    }

    if(!mDevice)
        return false;

    mManufacturer = QString::fromStdString(mDevice->getVendor());
    mModel = QString::fromStdString(mDevice->getModel());
    mSerial = QString::fromStdString(mDevice->getSerialNumber());

    mDevice->open(rcg::Device::CONTROL);
    std::shared_ptr<CNodeMapRef> nodeMap = mDevice->getRemoteNodeMap();

    CIntegerPtr ptrInt = nodeMap->_GetNode("Width");
    if(IsAvailable(ptrInt))
    {
        mWidth = ptrInt->GetMax();
    }

    ptrInt = nodeMap->_GetNode("Height");
    if (IsAvailable(ptrInt))
    {
        mHeight =  ptrInt->GetMax();
    }

    //Look for available pixel formats
    CEnumerationPtr ptrPixelFormats = nodeMap->_GetNode("PixelFormat");
    if(IsAvailable(ptrPixelFormats))
    {

        StringList_t names;
        ptrPixelFormats->GetSymbolics(names);
        QVector<int64_t> pixelFormats;

        for(const auto & name : names)
        {
            IEnumEntry* entry = ptrPixelFormats->GetEntryByName(name);
            if(entry)
                pixelFormats << entry->GetValue();
        }

        //BayerRG >> RGGB
        //BayerGR >> GRBG
        //BayerGB >> GBRG
        //BayerBG >> BGGR
        GenICam::gcstring fmtString;

        //12 bit packed
        if(pixelFormats.contains(BayerRG12p))
        {
            mImageFormat = cif12bpp_p;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_RGGB;
            fmtString = "BayerRG12p";
            mWhite = 4095;
            mIsColor = true;
        }
        else if(pixelFormats.contains(BayerGB12p))
        {
            mImageFormat = cif12bpp_p;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_GBRG;
            fmtString = "BayerGB12p";
            mWhite = 4095;
            mIsColor = true;
        }
        else if(pixelFormats.contains(BayerGR12p))
        {
            mImageFormat = cif12bpp_p;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_GRBG;
            fmtString = "BayerGR12p";
            mWhite = 4095;
            mIsColor = true;
        }
        else if(pixelFormats.contains(BayerBG12p))
        {
            mImageFormat = cif12bpp_p;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_BGGR;
            fmtString = "BayerBG12p";
            mWhite = 4095;
            mIsColor = true;
        }
        else if(pixelFormats.contains(Mono12p))
        {
            mImageFormat = cif12bpp_p;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_NONE;
            fmtString = "Mono12p";
            mWhite = 4095;
            mIsColor = false;
        }

        //12 bit unpacked
        else if(pixelFormats.contains(BayerRG12))
        {
            mImageFormat = cif12bpp;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_RGGB;
            fmtString = "BayerRG12";
            mWhite = 4095;
            mIsColor = true;
        }
        else if(pixelFormats.contains(BayerGB12))
        {
            mImageFormat = cif12bpp;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_GBRG;
            fmtString = "BayerGB12";
            mWhite = 4095;
            mIsColor = true;
        }
        else if(pixelFormats.contains(BayerGR12))
        {
            mImageFormat = cif12bpp;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_GRBG;
            fmtString = "BayerGR12";
            mWhite = 4095;
            mIsColor = true;
        }
        else if(pixelFormats.contains(BayerBG12))
        {
            mImageFormat = cif12bpp;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_BGGR;
            fmtString = "BayerBG12";
            mWhite = 4095;
            mIsColor = true;
        }
        else if(pixelFormats.contains(Mono12))
        {
            mImageFormat = cif12bpp;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_NONE;
            fmtString = "Mono12";
            mWhite = 4095;
            mIsColor = false;
        }

        //8 bit
        else if(pixelFormats.contains(BayerRG8))
        {
            mImageFormat = cif8bpp;
            mSurfaceFormat = FAST_I8;
            mPattern = FAST_BAYER_RGGB;
            fmtString = "BayerRG8";
            mWhite = 255;
            mIsColor = true;
        }
        else if(pixelFormats.contains(BayerGB8))
        {
            mImageFormat = cif8bpp;
            mSurfaceFormat = FAST_I8;
            mPattern = FAST_BAYER_GBRG;
            fmtString = "BayerGB8";
            mWhite = 255;
            mIsColor = true;
        }
        else if(pixelFormats.contains(BayerGR8))
        {
            mImageFormat = cif8bpp;
            mSurfaceFormat = FAST_I8;
            mPattern = FAST_BAYER_GRBG;
            fmtString = "BayerGR8";
            mWhite = 255;
            mIsColor = true;
        }
        else if(pixelFormats.contains(BayerBG8))
        {
            mImageFormat = cif8bpp;
            mSurfaceFormat = FAST_I8;
            mPattern = FAST_BAYER_BGGR;
            fmtString = "BayerBG8";
            mWhite = 255;
            mIsColor = true;
        }
        else if(pixelFormats.contains(Mono8))
        {
            mImageFormat = cif8bpp;
            mSurfaceFormat = FAST_I8;
            mPattern = FAST_BAYER_NONE;
            fmtString = "Mono8";
            mWhite = 255;
            mIsColor = false;
        }

        if (IsWritable(ptrPixelFormats))
        {
            // Retrieve the desired entry node from the enumeration node
            CEnumEntryPtr ptrPixFmt = ptrPixelFormats->GetEntryByName(fmtString);
            if (IsAvailable(ptrPixFmt) && IsReadable(ptrPixFmt))
            {
                // Retrieve the integer value from the entry node
                int64_t nPixelFormat = ptrPixFmt->GetValue();
                // Set integer as new value for enumeration node
                ptrPixelFormats->SetIntValue(nPixelFormat);
            }
        }
    }

    CBooleanPtr ptrAcquisitionFrameRateEnable = nodeMap->_GetNode("AcquisitionFrameRateEnable");
    if (IsAvailable(ptrAcquisitionFrameRateEnable) && IsWritable(ptrAcquisitionFrameRateEnable))
    {
        ptrAcquisitionFrameRateEnable->SetValue(true);
    }

    CFloatPtr ptrFloat = nodeMap->_GetNode("AcquisitionFrameRate");
    if(IsAvailable(ptrInt))
    {
        mFPS =  ptrFloat->GetValue();
    }

    CEnumerationPtr ptrLineSel = nodeMap->_GetNode("LineSelector");
    if(IsAvailable(ptrLineSel) && IsWritable(ptrLineSel))
    {
         CEnumEntryPtr ptrLineSelVal = ptrLineSel->GetEntryByName("Line2");
         ptrLineSel->SetIntValue(ptrLineSelVal->GetValue());

    }

    CEnumerationPtr ptrLineMode = nodeMap->_GetNode("LineMode");
    if(IsAvailable(ptrLineMode) && IsWritable(ptrLineMode))
    {
         CEnumEntryPtr ptrLineModeOut = ptrLineMode->GetEntryByName("Output");
         ptrLineMode->SetIntValue(ptrLineModeOut->GetValue());
    }

//    camera.LineSelector.SetValue(LineSelector_Line2);
//    camera.LineMode.SetValue(LineMode_Output);
//    LineModeEnums e = camera.LineMode.GetValue();



    if(!mInputBuffer.allocate(mWidth, mHeight, mSurfaceFormat))
        return false;


    mDevID = devID;
    mState = cstStopped;
    emit stateChanged(cstStopped);
    return true;
}

bool GeniCamCamera::start()
{
    mState = cstStreaming;
    emit stateChanged(cstStreaming);
    QTimer::singleShot(0, this, [this](){startStreaming();});
    return true;
}

bool GeniCamCamera::stop()
{
    mState = cstStopped;
    emit stateChanged(cstStopped);

    return true;
}

void GeniCamCamera::close()
{
    stop();
    mDevice->close();
    mState = cstClosed;
    emit stateChanged(cstClosed);
}

void GeniCamCamera::startStreaming()
{
    if(mState != cstStreaming)
        return;

    std::vector<std::shared_ptr<rcg::Stream>> streams = mDevice->getStreams();
    if(streams.empty())
        return;
    streams[0]->open();
    streams[0]->startStreaming();
    QElapsedTimer tmr;

    // Reset the camera statistics
    UpdateStatistics(nullptr);

    while(mState == cstStreaming)
    {
        tmr.restart();
        const rcg::Buffer* buffer = streams[0]->grab(3000);
        if(buffer == nullptr)
            continue;

        UpdateStatistics(buffer);

        if(buffer->getIsIncomplete())
            continue;

        //Multy part images not supported
        uint32_t npart = buffer->getNumberOfParts();
        if(npart > 1)
            continue;

        //if(buffer->getImagePresent(1))
        {
            const unsigned char* in = static_cast<const unsigned char *>(buffer->getBase(1));
            unsigned char* out = mInputBuffer.getBuffer();
            size_t sz = buffer->getSize(1);
            cudaMemcpy(out, in, sz, cudaMemcpyHostToDevice);
            mInputBuffer.release();
        }

        {
            QMutexLocker l(&mLock);
            mRawProc->acqTimeNsec = tmr.nsecsElapsed();
            mRawProc->wake();
        }
    }

    streams[0]->stopStreaming();
    streams[0]->close();
}

bool GeniCamCamera::getParameter(cmrCameraParameter param, float& val)
{
    using namespace GenApi;

    if(param < 0 || param > prmLast)
        return false;

    if(!mDevice)
        return false;

    CFloatPtr ptrFloat;
    CIntegerPtr ptrInt;
    std::shared_ptr<CNodeMapRef> nodeMap = mDevice->getRemoteNodeMap();
    switch (param)
    {
    case prmFrameRate:
        ptrFloat = nodeMap->_GetNode("AcquisitionFrameRate");
        if (IsAvailable(ptrFloat))
        {
            val = (float)ptrFloat->GetValue();
            return true;
        }
        else
        {
            ptrInt = nodeMap->_GetNode("AcquisitionFrameRate");
            if(IsAvailable(ptrInt))
            {
                val = (float)ptrInt->GetValue();
                return true;
            }
            else
            {
                return false;
            }
        }
        break;


    case prmExposureTime:
        ptrFloat = nodeMap->_GetNode("ExposureTime");
        if (IsAvailable(ptrFloat))
        {
            val = (float)ptrFloat->GetValue();
            return true;
        }
        else
        {
            ptrInt = nodeMap->_GetNode("ExposureTime");
            if(IsAvailable(ptrInt))
            {
                val = (float)ptrInt->GetValue();
                return true;
            }
            else
            {
                return false;
            }
        }

        break;

    default:
        break;
    }

    return false;
}

bool GeniCamCamera::setParameter(cmrCameraParameter param, float val)
{
    using namespace GenApi;

    if(param < 0 || param > prmLast)
        return false;

    if(!mDevice)
        return false;

    try
    {

        CFloatPtr ptrFloat;
        CIntegerPtr ptrInt;
        CBooleanPtr ptrAcquisitionFrameRateEnable;
        std::shared_ptr<CNodeMapRef> nodeMap = mDevice->getRemoteNodeMap();
        switch (param)
        {
        case prmFrameRate:
            ptrFloat = nodeMap->_GetNode("AcquisitionFrameRate");
            ptrAcquisitionFrameRateEnable = nodeMap->_GetNode("AcquisitionFrameRateEnable");
            if(IsAvailable(ptrAcquisitionFrameRateEnable) && IsWritable(ptrAcquisitionFrameRateEnable))
            {
                ptrAcquisitionFrameRateEnable->SetValue(true);
            }

            if (IsAvailable(ptrFloat) && IsWritable(ptrFloat))
            {
                ptrFloat->SetValue(val);
                return true;
            }
            else
            {
                ptrInt = nodeMap->_GetNode("AcquisitionFrameRate");
                if(IsAvailable(ptrInt) && IsWritable(ptrInt))
                {
                    ptrInt->SetValue(val);
                    return true;
                }
            }
            return false;

        case prmExposureTime:
        {
            // Set ExposureMode=Timed to make ExposureTime writable
            CEnumerationPtr ptrExpMode = nodeMap->_GetNode("ExposureMode");
            if(IsAvailable(ptrExpMode) && IsWritable(ptrExpMode))
                ptrExpMode->FromString("Timed");

            ptrFloat = nodeMap->_GetNode("ExposureTime");
            if (IsAvailable(ptrFloat))
            {
                if(IsWritable(ptrFloat))
                {
                    ptrFloat->SetValue(val);
                    return true;
                }
            }
            else
            {
                ptrInt = nodeMap->_GetNode("ExposureTime");
                if(IsAvailable(ptrInt))
                {
                    if(IsWritable(ptrInt))
                    {
                        ptrInt->SetValue(val);
                        return true;
                    }
                }
            }
            return false;
        }
        default:
            break;
        }
    }
    catch(GenICam::GenericException &ex)
    {
        // Show error here
        std::cout << "GenericException: " << ex.GetDescription() << std::endl;
    }

    return false;
}

bool GeniCamCamera::getParameterInfo(cmrParameterInfo& info)
{
    using namespace GenApi;

    if(info.param < 0 || info.param > prmLast)
        return false;

    if(!mDevice)
        return false;

    try {

        CFloatPtr ptrFloat;
        CIntegerPtr ptrInt;
        std::shared_ptr<CNodeMapRef> nodeMap = mDevice->getRemoteNodeMap();
        switch (info.param)
        {
        case prmFrameRate:
            ptrFloat = nodeMap->_GetNode("AcquisitionFrameRate");
            if (IsAvailable(ptrFloat))
            {
                info.min = (float)ptrFloat->GetMin();
                info.max = (float)ptrFloat->GetMax();
                if(ptrFloat->HasInc())
                    info.increment = (float)ptrFloat->GetInc();
                else
                    info.increment = 1;

                return true;
            }
            else
            {
                ptrInt = nodeMap->_GetNode("AcquisitionFrameRate");
                if(IsAvailable(ptrInt))
                {
                    info.min = (float)ptrInt->GetMin();
                    info.max = (float)ptrInt->GetMax();
                    info.increment = (float)ptrInt->GetInc();
                    return true;
                }
                else
                {
                    return false;
                }
            }

        case prmExposureTime:
        {
            // Set ExposureMode=Timed to make ExposureTime writable
            CEnumerationPtr ptrExpMode = nodeMap->_GetNode("ExposureMode");
            if(IsAvailable(ptrExpMode) && IsWritable(ptrExpMode))
                ptrExpMode->FromString("Timed");

            ptrFloat = nodeMap->_GetNode("ExposureTime");
            if (IsAvailable(ptrFloat))
            {
                info.min = (float)ptrFloat->GetMin();
                info.max = (float)ptrFloat->GetMax();
                if(ptrFloat->HasInc())
                    info.increment = (float)ptrFloat->GetInc();
                else
                    info.increment = 10;
                return true;
            }
            else
            {
                ptrInt = nodeMap->_GetNode("ExposureTime");
                if(IsAvailable(ptrInt))
                {
                    info.min = (float)ptrInt->GetMin();
                    info.max = (float)ptrInt->GetMax();
                    info.increment = (float)ptrInt->GetInc();
                }
                else
                {
                    return false;
                }
            }
            break;
        }
        default:
            break;
        }
    }
    catch(GenICam::GenericException &ex)
    {
        // Show error here
        std::cout << "GenericException: " << ex.GetDescription() << std::endl;
    }
    return false;
}

void GeniCamCamera::UpdateStatistics(const rcg::Buffer*  pBuff)
{
    if(pBuff)
    {
        if(pBuff->getIsIncomplete())
        {
            // Update statFramesIncomplete statistics
            mStatistics[CameraStatEnum::statFramesIncomplete]++;
        }
        else
        {
            // Update statistics
            // Total number of frames
            mStatistics[CameraStatEnum::statFramesTotal]++;
            // Timestamp (we do not know the frequency!)
            mStatistics[CameraStatEnum::statCurrTimestamp] = pBuff->getTimestamp();

            // new frame ID
            uint64_t newFrameId = pBuff->getFrameID();
            mStatistics[CameraStatEnum::statCurrFrameID] = newFrameId;

            // Check if frames were dropped in the Camera
            if(mCurrFrameID!=0 && mCurrFrameID+1!=newFrameId)
            {
                mDropFramesNum+=(newFrameId>mCurrFrameID)?(newFrameId-mCurrFrameID-1):0;
                mStatistics[CameraStatEnum::statFramesDropped] = mDropFramesNum;
            }
            mCurrFrameID = newFrameId;

            // calculate FPC and thoughput
            if(mFirstFrameTime.time_since_epoch().count()==0)
            {
                mFirstFrameTime = mPrevFrameTime = std::chrono::system_clock::now();
            }
            else
            {
                const bool bAverage = false; // !!! Set to true to caclulate average FPS and Thoughput, false for instant values
                if(bAverage)
                {
                    // total time between the first acquired frame and the current one;
                    auto micorsecTotal = std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::system_clock::now()-mFirstFrameTime);
                    auto usT = micorsecTotal.count();

                    // FPS - average per grabbing session
                    mStatistics[CameraStatEnum::statCurrFps100] = ((mStatistics[CameraStatEnum::statFramesTotal]-1)*100000000)/usT;

                    // Throughput
                    mTotalBytesTransferred+=pBuff->getDataSize();
                    mStatistics[CameraStatEnum::statCurrTroughputMbs100] = (mTotalBytesTransferred*800)/usT;
                }
                else
                {
                    // time between the previous frame and the current one;
                    auto currTime = std::chrono::system_clock::now();
                    auto micorsecTotal = std::chrono::duration_cast<std::chrono::microseconds> (currTime-mPrevFrameTime);
                    auto usT = micorsecTotal.count();

                    // FPS - average per grabbing session
                    mStatistics[CameraStatEnum::statCurrFps100] = 100000000/usT;

                    // Throughput
                    auto bytesTransferred=pBuff->getDataSize();
                    mStatistics[CameraStatEnum::statCurrTroughputMbs100] = (bytesTransferred*800)/usT;

                    mPrevFrameTime = currTime;
                }
            }

        }
    }
    else
    {
        // Reset the statistics
        mStatistics[CameraStatEnum::statCurrFps100] = 0;
        mStatistics[CameraStatEnum::statCurrFrameID] =0;
        mStatistics[CameraStatEnum::statCurrTimestamp] = 0;
        mStatistics[CameraStatEnum::statCurrTroughputMbs100] = 0;
        mStatistics[CameraStatEnum::statFramesDropped] = 0;
        mStatistics[CameraStatEnum::statFramesIncomplete] = 0;
        mStatistics[CameraStatEnum::statFramesTotal] = 0;

        mCurrFrameID = 0;
        mDropFramesNum = 0;
        mFirstFrameTime = kZeroTime;
        mPrevFrameTime = kZeroTime;
        mTotalBytesTransferred=0;
    }
};
#endif
