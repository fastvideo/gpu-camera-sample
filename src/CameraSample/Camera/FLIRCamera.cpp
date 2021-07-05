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

#ifdef SUPPORT_FLIR
#include "FLIRCamera.h"

#include "MainWindow.h"
#include "RawProcessor.h"

#include <QTimer>
#include <QElapsedTimer>
#include <QDebug>

FLIRCamera::FLIRCamera()
{
    try
    {
        mSystem = System::GetInstance();
        mCamList = mSystem->GetCameras();
        const unsigned int numCameras = mCamList.GetSize();
        if (numCameras == 0)
        {
            // Clear camera list before releasing system
            mCamList.Clear();

            // Release system
            mSystem->ReleaseInstance();
            return;
        }

        mCam = mCamList.GetByIndex(0);
    }
    catch (Spinnaker::Exception& e)
    {
        qDebug() << "Error: " << e.what();
        return;
    }

}

FLIRCamera::~FLIRCamera()
{
    mCam = nullptr;

    // Clear camera list before releasing system
    mCamList.Clear();

    // Release system
    mSystem->ReleaseInstance();
}

bool FLIRCamera::open(uint32_t devID)
{
    using namespace GenApi;

    mManufacturer = QStringLiteral("FLIR");
    if(mCam == nullptr)
        return false;

    // Initialize camera
    mCam->Init();


    INodeMap& nodeMap = mCam->GetNodeMap();
    INodeMap& nodeMapTLDevice = mCam->GetTLDeviceNodeMap();
    CStringPtr ptrStringSerial = nodeMapTLDevice.GetNode("DeviceSerialNumber");
    GenICam::gcstring deviceSerialNumber("");
    if(IsAvailable(ptrStringSerial) && IsReadable(ptrStringSerial))
    {
        deviceSerialNumber = ptrStringSerial->GetValue();
        mSerial = QString::fromLatin1(deviceSerialNumber.c_str(), (int)deviceSerialNumber.length());
    }


    CIntegerPtr ptrInt = nodeMap.GetNode("Width");
    if(IsAvailable(ptrInt))
    {
        mWidth = ptrInt->GetMax();
    }

    ptrInt = nodeMap.GetNode("Height");
    if (IsAvailable(ptrInt))
    {
        mHeight =  ptrInt->GetMax();
    }

    mSurfaceFormat = FAST_I8;
    mImageFormat = cif8bpp;
    //Look for available pixel formats
    CEnumerationPtr ptrPixelFormats = nodeMap.GetNode("PixelFormat");
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

    mFPS  = 0;

    CBooleanPtr ptrAcquisitionFrameRateEnable = nodeMap.GetNode("AcquisitionFrameRateEnable");
    if (IsAvailable(ptrAcquisitionFrameRateEnable) && IsWritable(ptrAcquisitionFrameRateEnable))
    {
        ptrAcquisitionFrameRateEnable->SetValue(true);
    }

    CFloatPtr ptrFloat = nodeMap.GetNode("AcquisitionFrameRate");
    if(IsAvailable(ptrInt))
    {
        mFPS =  ptrFloat->GetValue();
    }

    if(!mInputBuffer.allocate(mWidth, mHeight, mSurfaceFormat))
        return false;

    mDevID = devID;
    mState = cstStopped;
    emit stateChanged(cstStopped);
    return true;
}

bool FLIRCamera::start()
{
    mState = cstStreaming;
    emit stateChanged(cstStreaming);
    QTimer::singleShot(0, this, [this](){startStreaming();});
    return true;
}

bool FLIRCamera::stop()
{
    mState = cstStopped;
    emit stateChanged(cstStopped);

    return true;
}

void FLIRCamera::close()
{
    stop();
    mCam->DeInit();
    mState = cstClosed;
    emit stateChanged(cstClosed);
}

void FLIRCamera::startStreaming()
{
    if(mState != cstStreaming)
        return;
    QElapsedTimer tmr;

    mCam->BeginAcquisition();

    while(mState == cstStreaming)
    {
        tmr.restart();
        try
        {
             // Retrieve next received image
             ImagePtr pResultImage = mCam->GetNextImage(1000);

             // Ensure image is complete
             if (pResultImage->IsIncomplete())
             {
                 // Retrieve and print the image status description
                 //qDebug() << "Image incomplete: " << Image::GetImageStatusDescription(pResultImage->GetImageStatus());
                 continue;
             }
             else
             {
                 unsigned char* dst = mInputBuffer.getBuffer();
                 void* src = pResultImage->GetData();
                 size_t sz = pResultImage->GetImageSize();
                 cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice);
                 mInputBuffer.release();
             }

             // Release image
             pResultImage->Release();

         }
         catch (Spinnaker::Exception& e)
         {
             qDebug() << "Error: " << e.what();
             break;
         }


        {
            QMutexLocker l(&mLock);
            mRawProc->acqTimeNsec = tmr.nsecsElapsed();
            mRawProc->wake();
        }
    }

    mCam->EndAcquisition();
}


bool FLIRCamera::getParameter(cmrCameraParameter param, float& val)
{
    using namespace GenApi;

    if(param < 0 || param > prmLast)
        return false;

    if(!mCam)
        return false;

    CFloatPtr ptrFloat;
    CIntegerPtr ptrInt;
    INodeMap& nodeMap = mCam->GetNodeMap();
    switch (param)
    {
    case prmFrameRate:
        ptrFloat = nodeMap.GetNode("AcquisitionFrameRate");
        if (IsAvailable(ptrFloat))
        {
            val = (float)ptrFloat->GetValue();
            return true;
        }
        else
        {
            ptrInt = nodeMap.GetNode("AcquisitionFrameRate");
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
        ptrFloat = nodeMap.GetNode("ExposureTime");
        if (IsAvailable(ptrFloat))
        {
            val = (float)ptrFloat->GetValue();
            return true;
        }
        else
        {
            ptrInt = nodeMap.GetNode("ExposureTime");
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

bool FLIRCamera::setParameter(cmrCameraParameter param, float val)
{
    using namespace GenApi;

    if(param < 0 || param > prmLast)
        return false;

    if(!mCam)
        return false;

    CFloatPtr ptrFloat;
    CIntegerPtr ptrInt;
    CBooleanPtr ptrAcquisitionFrameRateEnable;
    INodeMap& nodeMap = mCam->GetNodeMap();
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

    default:
        break;
    }

    return false;
}

bool FLIRCamera::getParameterInfo(cmrParameterInfo& info)
{
    using namespace GenApi;

    if(info.param < 0 || info.param > prmLast)
        return false;

    if(!mCam)
        return false;

    CFloatPtr ptrFloat;
    CIntegerPtr ptrInt;
    INodeMap& nodeMap = mCam->GetNodeMap();
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

    default:
        break;
    }

    return false;
}

#endif
