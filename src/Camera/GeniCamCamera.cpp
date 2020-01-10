#include "GeniCamCamera.h"

#ifdef SUPPORT_GENICAM

#include <QVector>
#include <QElapsedTimer>

#include <RawProcessor.h>

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

    CFloatPtr ptrFloat = nodeMap->_GetNode("AcquisitionFrameRate");
    if(IsAvailable(ptrInt))
    {
        mFPS =  ptrFloat->GetValue();
    }

    //    ptrFloat = nodeMap->_GetNode("ExposureTime");
    //    if (IsAvailable(ptrFloat))
    //    {
    //        if(IsWritable(ptrFloat))
    //            ptrFloat->SetValue(100);
    //    }
    //    else
    //    {
    //        ptrInt = nodeMap->_GetNode("ExposureTime");
    //        if(IsAvailable(ptrInt))
    //        {
    //            if(IsWritable(ptrInt))
    //                ptrInt->SetValue(100);
    //        }
    //    }

    mInputBuffer.allocate(mWidth, mHeight, mSurfaceFormat);

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
    while(mState == cstStreaming)
    {
        tmr.restart();
        const rcg::Buffer* buffer = streams[0]->grab(3000);
        if(buffer == nullptr)
            continue;

        if(buffer->getIsIncomplete())
            continue;

        //Multy part images not supported
        uint32_t npart = buffer->getNumberOfParts();
        if(npart > 1)
            continue;

        if(buffer->getImagePresent(1))
        {
//            size_t width = buffer->getWidth(1);
//            size_t height = buffer->getHeight(1);
            const unsigned char* in = static_cast<const unsigned char *>(buffer->getBase(1));
            unsigned char* out = mInputBuffer.getBuffer();
            size_t sz = buffer->getSize(1);
            memcpy(out, in, sz);
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

//    xiStopAcquisition(hDevice);
}

#endif
