#include "LucidCamera.h"

#ifdef SUPPORT_LUCID
#include <QVector>
#include <QElapsedTimer>

#include <RawProcessor.h>

using CameraStatEnum = GPUCameraBase::cmrCameraStatistic;

LucidCamera::LucidCamera()
{
    mCameraThread.setObjectName(QStringLiteral("GeniCamThread"));
    moveToThread(&mCameraThread);
    mCameraThread.start();
}

LucidCamera::~LucidCamera()
{
    mCameraThread.quit();
    mCameraThread.wait(3000);

    close();
}

bool LucidCamera::open(uint32_t devID)
{
    using namespace GenApi;
    std::vector<Arena::DeviceInfo> deviceInfos;
    Arena::DeviceInfo devInfo;
    try
    {
        mSystem = Arena::OpenSystem();
        mSystem->UpdateDevices(100);
        deviceInfos = mSystem->GetDevices();
        if (deviceInfos.size() == 0)
        {
            qDebug() << "No camera connected";
            return false;
        }

        devInfo = deviceInfos[0];
        mDevice = mSystem->CreateDevice(devInfo);



        if(!mDevice)
            return false;

        mManufacturer = QString::fromStdString(devInfo.VendorName().c_str());
        mModel = QString::fromStdString(devInfo.ModelName().c_str());
        mSerial = QString::fromStdString(devInfo.SerialNumber().c_str());

        INodeMap* nodeMap = mDevice->GetNodeMap();

        CIntegerPtr ptrInt = nodeMap->GetNode("Width");
        if(IsAvailable(ptrInt))
        {
            mWidth = ptrInt->GetMax();
        }

        ptrInt = nodeMap->GetNode("Height");
        if (IsAvailable(ptrInt))
        {
            mHeight =  ptrInt->GetMax();
        }

        //Look for available pixel formats
        CEnumerationPtr ptrPixelFormats = nodeMap->GetNode("PixelFormat");
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

            Arena::SetNodeValue<GenICam::gcstring>(nodeMap, "PixelFormat", fmtString);

    //        if(IsWritable(ptrPixelFormats))
    //        {
    //            // Retrieve the desired entry node from the enumeration node
    //            CEnumEntryPtr ptrPixFmt = ptrPixelFormats->GetEntryByName(fmtString);
    //            if (IsAvailable(ptrPixFmt) && IsReadable(ptrPixFmt))
    //            {
    //                // Retrieve the integer value from the entry node
    //                int64_t nPixelFormat = ptrPixFmt->GetValue();
    //                // Set integer as new value for enumeration node
    //                ptrPixelFormats->SetIntValue(nPixelFormat);
    //            }
    //        }
        }



        CBooleanPtr ptrAcquisitionFrameRateEnable = nodeMap->GetNode("AcquisitionFrameRateEnable");
        if (IsAvailable(ptrAcquisitionFrameRateEnable) && IsWritable(ptrAcquisitionFrameRateEnable))
        {
            ptrAcquisitionFrameRateEnable->SetValue(true);
        }

        CFloatPtr ptrFloat = nodeMap->GetNode("AcquisitionFrameRate");
        if(IsAvailable(ptrInt))
        {
            mFPS =  ptrFloat->GetValue();
        }

        if(!mInputBuffer.allocate(mWidth, mHeight, mSurfaceFormat))
            return false;

        }
    catch (GenICam::GenericException& ge)
    {
        qDebug() << "GenICam exception thrown: " << ge.what();
        return false;
    }
    catch (std::exception& ex)
    {
        qDebug() << "Standard exception thrown: " << ex.what();
        return false;
    }
    catch (...)
    {
        qDebug() << "Unexpected exception thrown";
        return false;
    }

    mDevID = devID;
    mState = cstStopped;
    emit stateChanged(cstStopped);
    return true;
}

bool LucidCamera::start()
{
    mState = cstStreaming;
    emit stateChanged(cstStreaming);
    QTimer::singleShot(0, this, [this](){startStreaming();});
    return true;
}

bool LucidCamera::stop()
{
    mState = cstStopped;
    emit stateChanged(cstStopped);

    return true;
}

void LucidCamera::close()
{
    stop();
    if(mSystem != nullptr)
    {
        if(mDevice = nullptr)
        {
            mSystem->DestroyDevice(mDevice);
            mDevice = nullptr;
        }

        Arena::CloseSystem(mSystem);
        mSystem = nullptr;
    }
    mState = cstClosed;
    emit stateChanged(cstClosed);
}

void LucidCamera::startStreaming()
{
    if(mState != cstStreaming)
        return;

    using namespace GenApi;
    GenICam::gcstring acquisitionModeInitial;
    try
    {
        // get node values that will be changed in order to return their values at the end
        acquisitionModeInitial = Arena::GetNodeValue<GenICam::gcstring>(mDevice->GetNodeMap(), "AcquisitionMode");

        // Set acquisition mode
        //    Set acquisition mode before starting the stream. Starting the stream
        //    requires the acquisition mode to be set beforehand. The acquisition
        //    mode controls the number of images a device acquires once the stream
        //    has been started. Setting the acquisition mode to 'Continuous' keeps
        //    the stream from stopping. This example returns the camera to its
        //    initial acquisition mode near the end of the example.
        Arena::SetNodeValue<GenICam::gcstring>(mDevice->GetNodeMap(), "AcquisitionMode", "Continuous");

        // Set buffer handling mode
        //    Set buffer handling mode before starting the stream. Starting the
        //    stream requires the buffer handling mode to be set beforehand. The
        //    buffer handling mode determines the order and behavior of buffers in
        //    the underlying stream engine. Setting the buffer handling mode to
        //    'NewestOnly' ensures the most recent image is delivered, even if it
        //    means skipping frames.
        Arena::SetNodeValue<GenICam::gcstring>(mDevice->GetTLStreamNodeMap(), "StreamBufferHandlingMode", "NewestOnly");

        // Some 10G higher bandwidth LUCID cameras support TCP streaming.  TCP protocol
        //    implements a reliable connection based stream at the hardware level eliminating
        //    the need for software based packet resend mechanism.  This next check is to see
        //    if TCP streaming is available on the camera, if we will enable it, oherwise we
        //    demonstrate how to enable auto packetsize negotiation and packet resend for
        //    traditional UDP pased GVSP stream.
        if (IsImplemented(mDevice->GetNodeMap()->GetNode("TCPEnable")))
        {
            // The TCPEnable node will tell the camera to use the TCP datastream engine.  When
            //    enabled on the camera Arena will switch to using the TCP datastream engine.
            //    There is no further necessary configuration though to achieve maximum throughput
            //    users may want to set the "DeviceLinkThroughputReserve" to 0 and also set the
            //    stream channel packet delay "GevSCPD" to 0.
            Arena::SetNodeValue<bool>(mDevice->GetNodeMap(), "TCPEnable", true);
        }
        else
        {
            // Enable stream auto negotiate packet size
            //    Setting the stream packet size is done before starting the stream.
            //    Setting the stream to automatically negotiate packet size instructs the
            //    camera to receive the largest packet size that the system will allow.
            //    This generally increases frame rate and results in fewer interrupts per
            //    image, thereby reducing CPU load on the host system. Ethernet settings
            //    may also be manually changed to allow for a larger packet size.

            Arena::SetNodeValue<bool>(mDevice->GetTLStreamNodeMap(), "StreamAutoNegotiatePacketSize", true);

            // Enable stream packet resend
            //    Enable stream packet resend before starting the stream. Images are sent
            //    from the camera to the host in packets using UDP protocol, which
            //    includes a header image number, packet number, and timestamp
            //    information. If a packet is missed while receiving an image, a packet
            //    resend is requested and this information is used to retrieve and
            //    redeliver the missing packet in the correct order.

            Arena::SetNodeValue<bool>(mDevice->GetTLStreamNodeMap(), "StreamPacketResendEnable", true);
        }

        mDevice->StartStream();
        QElapsedTimer tmr;

        // Reset the camera statistics
        UpdateStatistics(nullptr);

        while(mState == cstStreaming)
        {
            tmr.restart();

            Arena::IImage* pImage = mDevice->GetImage(3000);

            UpdateStatistics(pImage);

            if(pImage->IsIncomplete())
                continue;

            //if(pImage->getImagePresent(1))
            {
                const unsigned char* in = static_cast<const unsigned char *>(pImage->GetData());
                unsigned char* out = mInputBuffer.getBuffer();
                size_t sz = pImage->GetSizeFilled();
                cudaMemcpy(out, in, sz, cudaMemcpyHostToDevice);
                mInputBuffer.release();
            }

            {
                QMutexLocker l(&mLock);
                mRawProc->acqTimeNsec = tmr.nsecsElapsed();
                mRawProc->wake();
            }
            mDevice->RequeueBuffer(pImage);
        }

        // Stop stream
        //    Stop the stream after all images have been requeued. Failing to stop
        //    the stream will leak memory.

        mDevice->StopStream();

        // return nodes to their initial values
        Arena::SetNodeValue<GenICam::gcstring>(mDevice->GetNodeMap(), "AcquisitionMode", acquisitionModeInitial);

    }
    catch (GenICam::GenericException& ge)
    {
        qDebug() << "GenICam exception thrown: " << ge.what();
        return;
    }
    catch (std::exception& ex)
    {
        qDebug() << "Standard exception thrown: " << ex.what();
        return;
    }
    catch (...)
    {
        qDebug() << "Unexpected exception thrown";
        return;
    }
}

bool LucidCamera::getParameter(cmrCameraParameter param, float& val)
{
    using namespace GenApi;

    if(param < 0 || param > prmLast)
        return false;

    if(!mDevice)
        return false;

    CFloatPtr ptrFloat;
    CIntegerPtr ptrInt;
    INodeMap* nodeMap = mDevice->GetNodeMap();
    switch (param)
    {
    case prmFrameRate:
        ptrFloat = nodeMap->GetNode("AcquisitionFrameRate");
        if (IsAvailable(ptrFloat))
        {
            val = (float)ptrFloat->GetValue();
            return true;
        }
        else
        {
            ptrInt = nodeMap->GetNode("AcquisitionFrameRate");
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
        ptrFloat = nodeMap->GetNode("ExposureTime");
        if (IsAvailable(ptrFloat))
        {
            val = (float)ptrFloat->GetValue();
            return true;
        }
        else
        {
            ptrInt = nodeMap->GetNode("ExposureTime");
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

bool LucidCamera::setParameter(cmrCameraParameter param, float val)
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
        INodeMap* nodeMap = mDevice->GetNodeMap();
        switch (param)
        {
        case prmFrameRate:
            ptrFloat = nodeMap->GetNode("AcquisitionFrameRate");
            ptrAcquisitionFrameRateEnable = nodeMap->GetNode("AcquisitionFrameRateEnable");
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
                ptrInt = nodeMap->GetNode("AcquisitionFrameRate");
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
            CEnumerationPtr ptrExpMode = nodeMap->GetNode("ExposureMode");
            if(IsAvailable(ptrExpMode) && IsWritable(ptrExpMode))
                ptrExpMode->FromString("Timed");

            ptrFloat = nodeMap->GetNode("ExposureTime");
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
                ptrInt = nodeMap->GetNode("ExposureTime");
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

bool LucidCamera::getParameterInfo(cmrParameterInfo& info)
{
    using namespace GenApi;

    if(info.param < 0 || info.param > prmLast)
        return false;

    if(!mDevice)
        return false;

    try {

        CFloatPtr ptrFloat;
        CIntegerPtr ptrInt;
        INodeMap* nodeMap = mDevice->GetNodeMap();
        switch (info.param)
        {
        case prmFrameRate:
            ptrFloat = nodeMap->GetNode("AcquisitionFrameRate");
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
                ptrInt = nodeMap->GetNode("AcquisitionFrameRate");
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
            CEnumerationPtr ptrExpMode = nodeMap->GetNode("ExposureMode");
            if(IsAvailable(ptrExpMode) && IsWritable(ptrExpMode))
                ptrExpMode->FromString("Timed");

            ptrFloat = nodeMap->GetNode("ExposureTime");
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
                ptrInt = nodeMap->GetNode("ExposureTime");
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

void LucidCamera::UpdateStatistics(Arena::IImage* pImage)
{
    if(pImage)
    {
        if(pImage->IsIncomplete())
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
            mStatistics[CameraStatEnum::statCurrTimestamp] = pImage->GetTimestamp();

            // new frame ID
            uint64_t newFrameId = pImage->GetFrameId();
            mStatistics[CameraStatEnum::statCurrFrameID] = newFrameId;

            // Check if frames were dropped in the Camera
            if(mCurrFrameID!=0 && mCurrFrameID+1!=newFrameId)
            {
                mDropFramesNum += (newFrameId > mCurrFrameID) ? (newFrameId - mCurrFrameID - 1) : 0;
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
                    mTotalBytesTransferred += pImage->GetSizeFilled();
                    mStatistics[CameraStatEnum::statCurrTroughputMbs100] = (mTotalBytesTransferred*800)/usT;
                }
                else
                {
                    // time between the previous frame and the current one;
                    auto currTime = std::chrono::system_clock::now();
                    auto micorsecTotal = std::chrono::duration_cast<std::chrono::microseconds> (currTime-mPrevFrameTime);
                    auto usT = micorsecTotal.count();

                    // FPS - average per grabbing session
                    mStatistics[CameraStatEnum::statCurrFps100] = 100000000 / usT;

                    // Throughput
                    auto bytesTransferred = pImage->GetSizeFilled();
                    mStatistics[CameraStatEnum::statCurrTroughputMbs100] = (bytesTransferred * 800) / usT;

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
