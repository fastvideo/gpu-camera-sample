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

#ifdef SUPPORT_IMPERX
#include "ImperxCamera.h"

#include "MainWindow.h"
#include "RawProcessor.h"

#include <QTimer>
#include <QElapsedTimer>
#include <QDebug>

using namespace IpxCam;
using CameraStatEnum = GPUCameraBase::cmrCameraStatistic  ;

ImperxCamera::ImperxCamera()
{
    mCameraThread.setObjectName(QStringLiteral("ImperxCameraThread"));
    moveToThread(&mCameraThread);
    mCameraThread.start();

    m_pSystem = IpxCam::IpxCam_GetSystem();

    int numCameras = 0;

    if(m_pSystem)
    {
        // Get List of Interfaces in the System
        auto intList = m_pSystem->GetInterfaceList();
        // Iterate the Interfaces
        for (auto iface = intList->GetFirst(); iface; iface = intList->GetNext())
        {
            // Get List of the Cameras (DevInfo!) on each Interface
            auto devInfoList = iface->GetDeviceInfoList();
            // Iterate the Cameras
            for (auto devInfo = devInfoList->GetFirst(); devInfo; devInfo = devInfoList->GetNext())
            {
                // Increase the cameras counter
                numCameras++;
            }
        }

        if (numCameras == 0)
        {
            // Release system
            m_pSystem->Release();

            //!!!!!! ??? throw the ERROR exceptiopn here
        }

    }

}

ImperxCamera::~ImperxCamera()
{
    mCameraThread.quit();
    mCameraThread.wait(3000);

    if(m_pSystem)
        m_pSystem->Release();
    m_pSystem=nullptr;

}

bool ImperxCamera::open(uint32_t devID)
{

    if(!m_pSystem)
        return false;

    uint32_t currIdx=0;
    // Get List of Interfaces in the System
    auto intList = m_pSystem->GetInterfaceList();
    // Iterate the Interfaces
    for (auto iface = intList->GetFirst(); iface; iface = intList->GetNext())
    {
        // Get List of the Cameras (DevInfo!) on each Interface
        auto devInfoList = iface->GetDeviceInfoList();
        // Iterate the Cameras
        for (auto devInfo = devInfoList->GetFirst(); devInfo; devInfo = devInfoList->GetNext(), ++currIdx)
        {
            // Create the camera with the specified index
            if(currIdx==devID)
            {
                m_pCamera = IpxCam::IpxCam_CreateDevice(devInfo);
                break;
            }
        }
        if(m_pCamera)
            break;
    }

    // Initialize the object from he camera
    if(m_pCamera)
    {
        IpxCamErr err=IPX_CAM_ERR_OK;

        // Parameters array
        m_pParameters = m_pCamera->GetCameraParameters();

        // Manufacturer
        mManufacturer = QStringLiteral("Imperx");

        // Serial Number
        auto paramSerialNum = m_pParameters->GetString("DeviceSerialNumber", &err);
        if(paramSerialNum && err==IPX_CAM_ERR_OK)
            mSerial = paramSerialNum->GetValue();

        // Width/Height
        mWidth = m_pParameters->GetIntegerValue("Width");
        mHeight = m_pParameters->GetIntegerValue("Height");

        // PixelFormat
        mSurfaceFormat = FAST_I8;
        mImageFormat = cif8bpp;

        PixFmt pixFormat = static_cast<PixFmt>(m_pParameters->GetEnumValue("PixelFormat"));

        switch(pixFormat)
        {
            case PixFmt::Mono8:
            {
                mImageFormat = cif8bpp;
                mSurfaceFormat = FAST_I8;
                mPattern = FAST_BAYER_NONE;
                mWhite = 255;
                mIsColor = false;
                break;
            }
            case PixFmt::Mono12:
            {
                mImageFormat = cif12bpp;
                mSurfaceFormat = FAST_I12;
                mPattern = FAST_BAYER_NONE;
                mWhite = 4095;
                mIsColor = false;
                break;
            }
            case PixFmt::BayerRG8:
            {
                mImageFormat = cif8bpp;
                mSurfaceFormat = FAST_I8;
                mPattern = FAST_BAYER_RGGB;
                mWhite = 255;
                mIsColor = true;
                break;
            }
            case PixFmt::BayerRG12:
            {
                mImageFormat = cif12bpp;
                mSurfaceFormat = FAST_I12;
                mPattern = FAST_BAYER_RGGB;
                mWhite = 4095;
                mIsColor = true;
                break;
            }
            default:
            {
                // ERROR!!! PixelFormat not implemented
                Q_ASSERT(0);
                break;
            }
        }

        // Find FPS
        mFPS = 1;
        m_pParameters->SetBooleanValue("AcquisitionFrameRateEnable", true);
        mFPS = m_pParameters->GetFloatValue("AcquisitionFrameRate");

        // Allocate image data buffer
        if(!mInputBuffer.allocate(mWidth, mHeight, mSurfaceFormat))
            return false;

        // Set Initial state
        mDevID = devID;
        mState = cstStopped;
        emit stateChanged(cstStopped);
    }


    // Return true, if camera was created successfully, false - otherwise
    return(m_pCamera != nullptr);

}

bool ImperxCamera::start()
{
    mState = cstStreaming;
    emit stateChanged(cstStreaming);
    QTimer::singleShot(0, this, [this](){
        startStreaming();
    });
    return true;
}

bool ImperxCamera::stop()
{
    mState = cstStopped;
    emit stateChanged(cstStopped);

    return true;
}

void ImperxCamera::close()
{
    stop();
    m_pCamera->Release();
    m_pCamera = nullptr;
    mState = cstClosed;
    emit stateChanged(cstClosed);
}

void ImperxCamera::startStreaming()
{
    if(mState != cstStreaming || m_pCamera==nullptr || m_pParameters ==nullptr )
        return;

    // Get stream object
    auto stream = m_pCamera->GetStreamByIndex();
    if(!stream)
        return;

     // Allocate buffers queue
    size_t buffQueueSize=10;
    if(IPX_CAM_ERR_OK != stream->AllocBufferQueue(nullptr, buffQueueSize))
    {
        stream->Release();
        return;
    }

    QElapsedTimer tmr;

    m_pParameters->SetIntegerValue("TLParamsLocked", 1);

    stream->StartAcquisition();

    m_pParameters->ExecuteCommand("AcquisitionStart");

    // Reset the Statistics
    UpdateStatistics(nullptr);

    while(mState == cstStreaming)
    {
        tmr.restart();

        // Retrieve next received image
        uint64_t timeout = 1000; // 1 second
        IpxCamErr err = IPX_CAM_ERR_OK;
        auto pBuff = stream->GetBuffer(timeout, &err);

        // Ensure image is complete
        if (pBuff)
        {
             // Update the Statistics
             UpdateStatistics(pBuff);

             if(pBuff->IsIncomplete())
             {
                 // Do not process incimplete buffers
                 stream->QueueBuffer(pBuff);
                 continue;
             }
             else
             {
                 // Process new acquired buffer
                 unsigned char* dst = mInputBuffer.getBuffer();
                 void* src = pBuff->GetBufferPtr();
                 size_t sz = pBuff->GetBufferSize();
                 cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice);
                 mInputBuffer.release();
                 stream->QueueBuffer(pBuff);
             }


        }

        {
            QMutexLocker l(&mLock);
            mRawProc->acqTimeNsec = tmr.nsecsElapsed();
            mRawProc->wake();
        }
    }

    m_pParameters->ExecuteCommand("AcquisitionStop");
    stream->StopAcquisition();
    stream->ReleaseBufferQueue();

    m_pParameters->SetIntegerValue("TLParamsLocked", 0);



}

bool ImperxCamera::getParameter(cmrCameraParameter param, float& val)
{
    if(param < 0 || param > prmLast || m_pCamera==nullptr || m_pParameters==nullptr )
        return false;

    switch (param)
    {
    case prmFrameRate:
    {
        auto currFT = m_pParameters->GetIntegerValue("CurrentFrameTime");
        val = (float)(1000000./float(currFT));
        return true;
    }
    case prmExposureTime:
    {
        auto currExp = m_pParameters->GetIntegerValue("CurrentExposure");
        val = (float)(currExp);
        return true;
    }
    default:
        break;
    }

    return false;
};

bool ImperxCamera::setParameter(cmrCameraParameter param, float val)
{

    if(param < 0 || param > prmLast || m_pCamera==nullptr || m_pParameters==nullptr )
        return false;

    IpxCamErr err= IPX_CAM_ERR_OK;

    switch (param)
    {
        case prmFrameRate:
        {
            err = m_pParameters->SetBooleanValue("AcquisitionFrameRateEnable", true);
            if(err== IPX_CAM_ERR_OK)
            {
                err = m_pParameters->SetFloatValue("AcquisitionFrameRate", val);
                if(err== IPX_CAM_ERR_OK)
                    mFPS = val;
            }

            return true;
        }
        case prmExposureTime:
        {
            err = m_pParameters->SetEnumValueStr("ExposureMode", "Timed");
            if(err== IPX_CAM_ERR_OK)
            {
                err = m_pParameters->SetFloatValue("ExposureTime", val);
                if(err== IPX_CAM_ERR_OK)
                    return true;
            }

            break;
        }
        default:
            break;
    }

    return false;
}

bool ImperxCamera::getParameterInfo(cmrParameterInfo& info)
{
    // Validate the input data and check if camera is properly connected
    if(info.param < 0 || info.param > prmLast || m_pCamera==nullptr || m_pParameters==nullptr )
        return false;

    IpxCamErr err= IPX_CAM_ERR_OK;

    switch (info.param)
    {
        case prmFrameRate:
        {
            auto paramFps = m_pParameters->GetFloat("AcquisitionFrameRate", &err);
            if(err == IPX_CAM_ERR_OK && paramFps)
            {
                info.min = paramFps->GetMin();
                info.max = paramFps->GetMax();
                info.increment = 1.;
                return true;
            }

            break;
        }
        case prmExposureTime:
        {
            auto paramExp = m_pParameters->GetFloat("ExposureTime", &err);
            if(err == IPX_CAM_ERR_OK && paramExp)
            {
                info.min = paramExp->GetMin();
                info.max = paramExp->GetMax();
                info.increment = 1.;
                return true;
            }

            break;
        }
        default:
            break;
    }

   return false;
}
void ImperxCamera::UpdateStatistics(IpxCam::Buffer *pBuff)
{
    if(pBuff)
    {
        if(pBuff->IsIncomplete())
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
            mStatistics[CameraStatEnum::statCurrTimestamp] = pBuff->GetTimestamp();

            // new frame ID
            uint64_t newFrameId = pBuff->GetFrameID();
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
                    mTotalBytesTransferred+=pBuff->GetBufferSize();
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
                    auto bytesTransferred=pBuff->GetBufferSize();
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
