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

#include "XimeaCamera.h"

#ifdef SUPPORT_XIMEA
#include "MainWindow.h"
#include "RawProcessor.h"

XimeaCamera::XimeaCamera() :
    CameraBase()
{
    mCameraThread.setObjectName(QStringLiteral("XimeaCameraThread"));
    moveToThread(&mCameraThread);
    mCameraThread.start();
}

XimeaCamera::~XimeaCamera()
{
    mCameraThread.quit();
    mCameraThread.wait(3000);
}
bool XimeaCamera::open(uint32_t devID)
{
    XI_RETURN ret = XI_OK;
    try
    {
        mManufacturer = QStringLiteral("Ximea");

        ret = xiOpenDevice(devID, &hDevice);
        if(ret != XI_OK)
        {
            qDebug("Cannot open camera %d, ret = %d", devID, ret);
            return false;
        }

        xiStopAcquisition(hDevice);

        char str[256] = {0};
        ret = xiGetParamString(hDevice, XI_PRM_DEVICE_NAME, str, sizeof(str));
        mModel = QString::fromLocal8Bit(str);

        ret = xiGetParamString(hDevice, XI_PRM_DEVICE_SN, str, sizeof(str));
        mSerial = QString::fromLocal8Bit(str);

        //Try to set 12 bit mode
        int image_data_bit_depth = 12;
        ret = xiSetParamInt(hDevice, XI_PRM_SENSOR_DATA_BIT_DEPTH, image_data_bit_depth);
        ret = xiGetParamInt(hDevice, XI_PRM_SENSOR_DATA_BIT_DEPTH, &image_data_bit_depth);

        if(image_data_bit_depth == XI_BPP_24 || image_data_bit_depth == XI_BPP_32)
        {
            ret = xiCloseDevice(hDevice);
            qDebug("24 and 32 bpp modes are not supported");
            return false;
        }

        if(image_data_bit_depth == XI_BPP_8)
            ret = xiSetParamInt(hDevice, XI_PRM_IMAGE_DATA_FORMAT, XI_RAW8);
        else
        {
            ret = xiSetParamInt(hDevice, XI_PRM_IMAGE_DATA_FORMAT, XI_FRM_TRANSPORT_DATA);
            ret = xiSetParamInt(hDevice, XI_PRM_OUTPUT_DATA_BIT_DEPTH, image_data_bit_depth);
            ret = xiSetParamInt(hDevice, XI_PRM_OUTPUT_DATA_PACKING, XI_ON);
        }

        int pack;
        xiGetParamInt(hDevice, XI_PRM_OUTPUT_DATA_PACKING, &pack);

        if(image_data_bit_depth == XI_BPP_8)
        {
            mImageFormat = cif8bpp;
            mSurfaceFormat = FAST_I8;
        }
        else if(image_data_bit_depth == XI_BPP_12 && pack == XI_ON)
        {
            mImageFormat = cif12bpp_p;
            mSurfaceFormat = FAST_I12;
        }
        else if(image_data_bit_depth == XI_BPP_12 && pack == XI_OFF)
        {
            mSurfaceFormat = FAST_I12;
            mImageFormat = cif12bpp;
        }
        else
        {
            mSurfaceFormat = FAST_I16;
            mImageFormat = cif16bpp;
        }

        mWhite = (1 << image_data_bit_depth) - 1;

        ret = xiSetParamInt(hDevice,XI_PRM_BUFFER_POLICY, XI_BP_SAFE);

        int is_color = 0;
        ret = xiGetParamInt(hDevice, XI_PRM_IMAGE_IS_COLOR, &is_color);

        ret = xiSetParamFloat(hDevice, XI_PRM_EXP_PRIORITY, 1);

        xiSetParamInt(hDevice, XI_PRM_ACQ_TIMING_MODE, XI_ACQ_TIMING_MODE_FREE_RUN);
        xiGetParamFloat(hDevice, XI_PRM_FRAMERATE, &mFPS);


        // color camera
        if(is_color)
        {
            int cfa = 0;
            ret = xiGetParamInt(hDevice, XI_PRM_COLOR_FILTER_ARRAY, &cfa);
            switch (cfa)
            {
            case XI_CFA_BAYER_RGGB:
                mPattern = FAST_BAYER_RGGB;
                break;
            case XI_CFA_BAYER_BGGR:
                mPattern = FAST_BAYER_BGGR;
                break;
            case XI_CFA_BAYER_GRBG:
                mPattern = FAST_BAYER_GRBG;
                break;
            case XI_CFA_BAYER_GBRG:
                mPattern = FAST_BAYER_GBRG;
                break;
            default:
                mPattern = FAST_BAYER_NONE;
                break;
            }
        }
        else
            mPattern = FAST_BAYER_NONE;

        ret = xiGetParamInt(hDevice, XI_PRM_WIDTH, &mWidth);
        ret = xiGetParamInt(hDevice, XI_PRM_HEIGHT, &mHeight);
    }
    catch(const char* err)
    {
        qDebug("Error: %s\n", err);
        return false;
    }

    mInputBuffer.allocate(mWidth, mHeight, mSurfaceFormat);

    mDevID = devID;
    mState = cstStopped;
    emit stateChanged(cstStopped);
    return true;
}

bool XimeaCamera::start()
{
    mState = cstStreaming;
    emit stateChanged(cstStreaming);
    QTimer::singleShot(0, this, [this](){startStreaming();});
    return true;
}

bool XimeaCamera::stop()
{
    mState = cstStopped;
    emit stateChanged(cstStopped);

    return true;
}

void XimeaCamera::close()
{
    stop();
    xiCloseDevice(hDevice);
    mState = cstClosed;
    emit stateChanged(cstClosed);
}

void XimeaCamera::startStreaming()
{
    if(mState != cstStreaming)
        return;
    XI_RETURN ret = XI_OK;

    ret = xiStartAcquisition(hDevice);
    if(ret != XI_OK)
        return;
    QElapsedTimer tmr;
    while(mState == cstStreaming)
    {
        tmr.restart();
        XI_IMG image = {0};
        image.size = sizeof(XI_IMG);
        image.bp_size = mInputBuffer.size();
        image.bp = mInputBuffer.getBuffer();
        ret = xiGetImage(hDevice, 5000, &image);
        mInputBuffer.release();

        {
            QMutexLocker l(&mLock);
            mRawProc->acqTimeNsec = tmr.nsecsElapsed();
            mRawProc->wake();
        }
    }
    xiStopAcquisition(hDevice);
}

#endif // SUPPORT_XIMEA
