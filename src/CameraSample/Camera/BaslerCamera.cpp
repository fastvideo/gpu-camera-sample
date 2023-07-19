#include "BaslerCamera.h"

#ifdef SUPPORT_BASLER
#include <QDebug>
#include <QElapsedTimer>

#include "RawProcessor.h"

BaslerCamera::BaslerCamera()
{
    // Namespace for using pylon objects.
    using namespace Pylon;

    // Before using any pylon methods, the pylon runtime must be initialized.
    PylonInitialize();

    mCameraThread.setObjectName(QStringLiteral("BaslerCamThread"));
    moveToThread(&mCameraThread);
    mCameraThread.start();
}

BaslerCamera::~BaslerCamera()
{
    // Namespace for using pylon objects.
    using namespace Pylon;

    mCameraThread.quit();
    mCameraThread.wait(3000);

    // Releases all pylon resources.
    PylonTerminate();
}

bool BaslerCamera::open(uint32_t devID)
{
    using namespace Pylon;
    using namespace GENAPI_NAMESPACE;
    bool res = true;
    try
    {
        // Create an instant camera object with the camera device found first.
        mCamera.reset(new CInstantCamera(CTlFactory::GetInstance().CreateFirstDevice()));
    }
    catch(const GenericException& e)
    {
        // Error handling.
        qDebug() << "An exception occurred." << e.GetDescription();
        res = false;
    }

    if(!res)
        return false;

    mManufacturer = QStringLiteral("Basler AG");
    mModel = QString::fromLatin1(mCamera->GetDeviceInfo().GetModelName().c_str());
    mSerial = QString::fromLatin1(mCamera->GetDeviceInfo().GetSerialNumber().c_str());

    mCamera->Open();

//    CIntegerParameter width(camera.GetNodeMap(), "Width");
//    camera.GetNodeMap()

    GenApi::INodeMap& nodemap = mCamera->GetNodeMap();

    CIntegerParameter p(nodemap, "Width");
    if(IsAvailable(p))
    {
        mWidth = p.GetValue();// .GetMax();
    }

    p = CIntegerParameter(nodemap, "Height");
    if(IsAvailable(p))
    {
        mHeight =  p.GetValue();// .GetMax();
    }

    //Look for available pixel formats
    CEnumParameter formats(nodemap, "PixelFormat");
    if(IsAvailable(formats))
    {
        StringList_t names;
        formats.GetSymbolics(names);
        QVector<int64_t> pixelFormats;

        for(const auto & name : names)
        {
            GenApi::IEnumEntry* entry = formats.GetEntryByName(name);
            if(entry)
                pixelFormats << entry->GetValue();
        }

        //BayerRG >> RGGB
        //BayerGR >> GRBG
        //BayerGB >> GBRG
        //BayerBG >> BGGR
        GenICam::gcstring fmtString;

        //12 bit packed
        if(pixelFormats.contains(EPixelType::PixelType_BayerRG12p))
        {
            mImageFormat = cif12bpp_p;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_RGGB;
            fmtString = "BayerRG12p";
            mWhite = 4095;
            mIsColor = true;
        }
        else if(pixelFormats.contains(EPixelType::PixelType_BayerGB12p))
        {
            mImageFormat = cif12bpp_p;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_GBRG;
            fmtString = "BayerGB12p";
            mWhite = 4095;
            mIsColor = true;
        }
        else if(pixelFormats.contains(EPixelType::PixelType_BayerGR12p))
        {
            mImageFormat = cif12bpp_p;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_GRBG;
            fmtString = "BayerGR12p";
            mWhite = 4095;
            mIsColor = true;
        }
        else if(pixelFormats.contains(EPixelType::PixelType_BayerBG12p))
        {
            mImageFormat = cif12bpp_p;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_BGGR;
            fmtString = "BayerBG12p";
            mWhite = 4095;
            mIsColor = true;
        }
//        else if(pixelFormats.contains(EPixelType::PixelType_Mono12p))
//        {
//            mImageFormat = cif12bpp_p;
//            mSurfaceFormat = FAST_I12;
//            mPattern = FAST_BAYER_NONE;
//            fmtString = "Mono12p";
//            mWhite = 4095;
//            mIsColor = false;
//        }

        //12 bit unpacked
        else if(pixelFormats.contains(EPixelType::PixelType_BayerRG12))
        {
            mImageFormat = cif12bpp;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_RGGB;
            fmtString = "BayerRG12";
            mWhite = 4095;
            mIsColor = true;
        }
        else if(pixelFormats.contains(EPixelType::PixelType_BayerGB12))
        {
            mImageFormat = cif12bpp;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_GBRG;
            fmtString = "BayerGB12";
            mWhite = 4095;
            mIsColor = true;
        }
        else if(pixelFormats.contains(EPixelType::PixelType_BayerGR12))
        {
            mImageFormat = cif12bpp;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_GRBG;
            fmtString = "BayerGR12";
            mWhite = 4095;
            mIsColor = true;
        }
        else if(pixelFormats.contains(EPixelType::PixelType_BayerBG12))
        {
            mImageFormat = cif12bpp;
            mSurfaceFormat = FAST_I12;
            mPattern = FAST_BAYER_BGGR;
            fmtString = "BayerBG12";
            mWhite = 4095;
            mIsColor = true;
        }
//        else if(pixelFormats.contains(EPixelType::PixelType_Mono12))
//        {
//            mImageFormat = cif12bpp;
//            mSurfaceFormat = FAST_I12;
//            mPattern = FAST_BAYER_NONE;
//            fmtString = "Mono12";
//            mWhite = 4095;
//            mIsColor = false;
//        }

        //8 bit
        else if(pixelFormats.contains(EPixelType::PixelType_BayerRG8))
        {
            mImageFormat = cif8bpp;
            mSurfaceFormat = FAST_I8;
            mPattern = FAST_BAYER_RGGB;
            fmtString = "BayerRG8";
            mWhite = 255;
            mIsColor = true;
        }
        else if(pixelFormats.contains(EPixelType::PixelType_BayerGB8))
        {
            mImageFormat = cif8bpp;
            mSurfaceFormat = FAST_I8;
            mPattern = FAST_BAYER_GBRG;
            fmtString = "BayerGB8";
            mWhite = 255;
            mIsColor = true;
        }
        else if(pixelFormats.contains(EPixelType::PixelType_BayerGR8))
        {
            mImageFormat = cif8bpp;
            mSurfaceFormat = FAST_I8;
            mPattern = FAST_BAYER_GRBG;
            fmtString = "BayerGR8";
            mWhite = 255;
            mIsColor = true;
        }
        else if(pixelFormats.contains(EPixelType::PixelType_BayerBG8))
        {
            mImageFormat = cif8bpp;
            mSurfaceFormat = FAST_I8;
            mPattern = FAST_BAYER_BGGR;
            fmtString = "BayerBG8";
            mWhite = 255;
            mIsColor = true;
        }
        else if(pixelFormats.contains(EPixelType::PixelType_Mono8))
        {
            mImageFormat = cif8bpp;
            mSurfaceFormat = FAST_I8;
            mPattern = FAST_BAYER_NONE;
            fmtString = "Mono8";
            mWhite = 255;
            mIsColor = false;
        }

        if(IsWritable(formats))
        {
            // Retrieve the desired entry node from the enumeration node
            CEnumEntryPtr ptrPixFmt = formats.GetEntryByName(fmtString);
            if(IsAvailable(ptrPixFmt) && IsReadable(ptrPixFmt))
            {
                // Retrieve the integer value from the entry node
                int64_t nPixelFormat = ptrPixFmt->GetValue();
                // Set integer as new value for enumeration node
                formats.SetIntValue(nPixelFormat);
            }
        }
    }

    CBooleanParameter acquisitionFrameRateEnable(nodemap, "AcquisitionFrameRateEnable");
    if (IsAvailable(acquisitionFrameRateEnable) && IsWritable(acquisitionFrameRateEnable))
    {
        acquisitionFrameRateEnable.SetValue(true);
    }

    CFloatParameter fps(nodemap, "AcquisitionFrameRate");
    if(IsAvailable(fps))
    {
        mFPS =  fps.GetValue();
    }



    CEnumParameter lineSel(nodemap, "LineSelector");
    if(IsAvailable(lineSel) && IsWritable(lineSel))
    {
         CEnumEntryPtr ptrLineSelVal = lineSel.GetEntryByName("Line2");
         if(IsAvailable(ptrLineSelVal))
         {
             lineSel.SetIntValue(ptrLineSelVal->GetValue());
         }

    }

    CBooleanParameter lineInv(nodemap, "LineInverter");
    if(IsAvailable(lineInv) && IsWritable(lineInv))
    {
        lineInv.SetValue(true);
    }

    CEnumParameter(nodemap, "LineSource").SetValue("ExposureActive");

    CEnumParameter lineMode(nodemap, "LineMode");
    if(IsAvailable(lineMode) && IsWritable(lineMode))
    {
         CEnumEntryPtr ptrLineModeOut = lineMode.GetEntryByName("Output");
         if(IsAvailable(ptrLineModeOut))
         {
             lineMode.SetIntValue(ptrLineModeOut->GetValue());
         }
    }





    if(!mInputBuffer.allocate(mWidth, mHeight, mSurfaceFormat))
        return false;

    mDevID = devID;
    mState = cstStopped;
    emit stateChanged(cstStopped);
    return true;
}

bool BaslerCamera::start()
{
    mState = cstStreaming;
    emit stateChanged(cstStreaming);
    QTimer::singleShot(0, this, [this](){startStreaming();});
    return true;
}

bool BaslerCamera::stop()
{
    mState = cstStopped;
    emit stateChanged(cstStopped);

    return true;
}

void BaslerCamera::close()
{
    stop();
    mCamera->Close();
    mState = cstClosed;
    emit stateChanged(cstClosed);
}

void BaslerCamera::startStreaming()
{
    using namespace Pylon;
    using namespace GENAPI_NAMESPACE;

    if(mState != cstStreaming)
        return;

    // Reset the camera statistics
    UpdateStatistics(CGrabResultPtr());

    CGrabResultPtr ptrGrabResult;

    mCamera->StartGrabbing();

//    uint64_t cnt = 0;
//    float expTime[] = {8000, 8000};

    while(mState == cstStreaming)
    {
//        tmr.restart();
        // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        mCamera->RetrieveResult(5000, ptrGrabResult, TimeoutHandling_Return);


        // Image grabbed successfully?
        if(!ptrGrabResult->GrabSucceeded())
            continue;

//        setParameter(prmExposureTime, expTime[cnt % 2]);
//        cnt++;

        UpdateStatistics(ptrGrabResult);

        const uint8_t* in = (uint8_t*) ptrGrabResult->GetBuffer();
        unsigned char* out = mInputBuffer.getBuffer();
        size_t sz = ptrGrabResult->GetImageSize();
        cudaMemcpy(out, in, sz, cudaMemcpyHostToDevice);
        mInputBuffer.release();

        mRawProc->wake();
    }
    mCamera->StopGrabbing();
}

bool BaslerCamera::getParameter(cmrCameraParameter param, float& val)
{
    using namespace Pylon;
    using namespace GENAPI_NAMESPACE;

    if(param < 0 || param > prmLast)
        return false;

    if(!mCamera)
        return false;

    CFloatParameter prFloat;
    CIntegerParameter prInt;
    GenApi::INodeMap& nodeMap = mCamera->GetNodeMap();
    switch (param)
    {
    case prmFrameRate:
        prFloat = CFloatParameter(nodeMap, "AcquisitionFrameRate");
        if(IsAvailable(prFloat))
        {
            val = (float)prFloat.GetValue();
            return true;
        }
        else
        {
            prInt = CIntegerParameter(nodeMap, "AcquisitionFrameRate");
            if(IsAvailable(prInt))
            {
                val = (float)prInt.GetValue();
                return true;
            }
            else
            {
                return false;
            }
        }
        break;


    case prmExposureTime:
        prFloat = CFloatParameter(nodeMap, "ExposureTime");
        if(IsAvailable(prFloat))
        {
            val = (float)prFloat.GetValue();
            return true;
        }
        else
        {
            prInt = CIntegerParameter(nodeMap, "ExposureTime");
            if(IsAvailable(prInt))
            {
                val = (float)prInt.GetValue();
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

bool BaslerCamera::setParameter(cmrCameraParameter param, float val)
{
    using namespace Pylon;
    using namespace GENAPI_NAMESPACE;

    if(param < 0 || param > prmLast)
        return false;

    if(!mCamera)
        return false;

    try
    {
        GenApi::INodeMap& nodeMap = mCamera->GetNodeMap();
        CFloatPtr ptrFloat;
        CIntegerPtr ptrInt;
        CBooleanPtr ptrAcquisitionFrameRateEnable;

        switch (param)
        {
        case prmFrameRate:
            ptrFloat = nodeMap.GetNode("AcquisitionFrameRate");
            ptrAcquisitionFrameRateEnable = nodeMap.GetNode("AcquisitionFrameRateEnable");
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
                ptrInt = nodeMap.GetNode("AcquisitionFrameRate");
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
            CEnumerationPtr ptrExpMode = nodeMap.GetNode("ExposureMode");
            if(IsAvailable(ptrExpMode) && IsWritable(ptrExpMode))
                ptrExpMode->FromString("Timed");

            ptrFloat = nodeMap.GetNode("ExposureTime");
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
                ptrInt = nodeMap.GetNode("ExposureTime");
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
        qDebug() << "GenericException: " << ex.GetDescription();
    }

    return false;
}

bool BaslerCamera::getParameterInfo(cmrParameterInfo& info)
{
    using namespace Pylon;
    using namespace GENAPI_NAMESPACE;

    if(info.param < 0 || info.param > prmLast)
        return false;

    if(!mCamera)
        return false;

    try {

        CFloatPtr ptrFloat;
        CIntegerPtr ptrInt;
        GenApi::INodeMap& nodeMap = mCamera->GetNodeMap();
        switch (info.param)
        {
        case prmFrameRate:
            ptrFloat = nodeMap.GetNode("AcquisitionFrameRate");
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
                ptrInt = nodeMap.GetNode("AcquisitionFrameRate");
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
            CEnumerationPtr ptrExpMode = nodeMap.GetNode("ExposureMode");
            if(IsAvailable(ptrExpMode) && IsWritable(ptrExpMode))
                ptrExpMode->FromString("Timed");

            ptrFloat = nodeMap.GetNode("ExposureTime");
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
                ptrInt = nodeMap.GetNode("ExposureTime");
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
        qDebug() << "GenericException: " << ex.GetDescription();
    }
    return false;
}

void BaslerCamera::UpdateStatistics(const Pylon::CGrabResultPtr pBuff)
{
    if(pBuff.IsValid())
    {
        if(!pBuff->GrabSucceeded())
        {
            // Update statFramesIncomplete statistics
            mStatistics[cmrCameraStatistic::statFramesIncomplete]++;
        }
        else
        {
            // Update statistics
            // Total number of frames
            mStatistics[cmrCameraStatistic::statFramesTotal]++;
            // Timestamp (we do not know the frequency!)
            mStatistics[cmrCameraStatistic::statCurrTimestamp] = pBuff->GetTimeStamp();

            // new frame ID
            uint64_t newFrameId = pBuff->GetID();
            mStatistics[cmrCameraStatistic::statCurrFrameID] = newFrameId;

            // Check if frames were dropped in the Camera
            if(mCurrFrameID != 0 && mCurrFrameID + 1 != newFrameId)
            {
                mDropFramesNum += (newFrameId > mCurrFrameID) ? (newFrameId - mCurrFrameID - 1):0;
                mStatistics[cmrCameraStatistic::statFramesDropped] = mDropFramesNum;
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
                    mStatistics[cmrCameraStatistic::statCurrFps100] = ((mStatistics[cmrCameraStatistic::statFramesTotal] - 1) * 100000000) / usT;

                    // Throughput
                    mTotalBytesTransferred += pBuff->GetImageSize();
                    mStatistics[cmrCameraStatistic::statCurrTroughputMbs100] = (mTotalBytesTransferred * 800) / usT;
                }
                else
                {
                    // time between the previous frame and the current one;
                    auto currTime = std::chrono::system_clock::now();
                    auto micorsecTotal = std::chrono::duration_cast<std::chrono::microseconds> (currTime-mPrevFrameTime);
                    auto usT = micorsecTotal.count();

                    // FPS - average per grabbing session
                    mStatistics[cmrCameraStatistic::statCurrFps100] = 100000000/usT;

                    // Throughput
                    auto bytesTransferred = pBuff->GetImageSize();
                    mStatistics[cmrCameraStatistic::statCurrTroughputMbs100] = (bytesTransferred*800)/usT;

                    mPrevFrameTime = currTime;
                }
            }

        }
    }
    else
    {
        // Reset the statistics
        mStatistics[cmrCameraStatistic::statCurrFps100] = 0;
        mStatistics[cmrCameraStatistic::statCurrFrameID] =0;
        mStatistics[cmrCameraStatistic::statCurrTimestamp] = 0;
        mStatistics[cmrCameraStatistic::statCurrTroughputMbs100] = 0;
        mStatistics[cmrCameraStatistic::statFramesDropped] = 0;
        mStatistics[cmrCameraStatistic::statFramesIncomplete] = 0;
        mStatistics[cmrCameraStatistic::statFramesTotal] = 0;

        mCurrFrameID = 0;
        mDropFramesNum = 0;
        mFirstFrameTime = kZeroTime;
        mPrevFrameTime = kZeroTime;
        mTotalBytesTransferred=0;
    }
}

#endif
