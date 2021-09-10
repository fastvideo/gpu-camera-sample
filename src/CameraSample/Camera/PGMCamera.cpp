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

#include "PGMCamera.h"
#include "MainWindow.h"
#include "RawProcessor.h"

extern int loadPPM(const char *file, void** data, BaseAllocator *alloc, unsigned int &width, unsigned &wPitch, unsigned int &height, unsigned &bitsPerPixel, unsigned &channels);

PGMCamera::PGMCamera(const QString &fileName,
                     fastBayerPattern_t  pattern,
                     bool isColor) :
    mFileName(fileName)
{
    mPattern = pattern;
    mIsColor = isColor;
    mCameraThread.setObjectName(QStringLiteral("PGMCameraThread"));
    moveToThread(&mCameraThread);
    mCameraThread.start();
}

PGMCamera::~PGMCamera()
{
    mCameraThread.quit();
    mCameraThread.wait(3000);
}

bool PGMCamera::open(uint32_t devID)
{
    Q_UNUSED(devID)

    mState = cstClosed;

    mManufacturer = QStringLiteral("Fastvideo");
    mModel = QStringLiteral("PGM camera simulator");
    mSerial = QStringLiteral("0000");

    FastAllocator a;
    unsigned char* bits = nullptr;
    uint width = 0;
    uint height = 0;
    uint pitch = 0;
    uint sampleSize = 0;
    uint samples = 0;
    if(1 != loadPPM(mFileName.toStdString().c_str(),
                    reinterpret_cast<void**>(&bits),
                    &a,
                    width, pitch, height,
                    sampleSize, samples))
        return false;

    if(samples != 1)
        return false;

    mFPS = 30;

    if(sampleSize == 8)
    {
        mImageFormat = cif8bpp;
        mSurfaceFormat = FAST_I8;
    }
    else if(sampleSize == 12)
    {
        mImageFormat = cif12bpp;
        mSurfaceFormat = FAST_I12;
    }
    else
    {
        mImageFormat = cif16bpp;
        mSurfaceFormat = FAST_I16;
    }

    mWidth = width;
    mHeight = height;
    mWhite = (1 << sampleSize) - 1;
    mBblack = 0;

    mInputImage.w = width;
    mInputImage.h = height;
    mInputImage.surfaceFmt = mSurfaceFormat;
    mInputImage.wPitch = pitch;
    mInputImage.bitsPerChannel = sampleSize;

    try
    {
        mInputImage.data.reset(static_cast<unsigned char*>(a.allocate(mInputImage.wPitch * mInputImage.h)));
    }
    catch(...)
    {
        return false;
    }

    memcpy(mInputImage.data.get(), bits, pitch * height);
    a.deallocate(bits);

    if(!mInputBuffer.allocate(mWidth, mHeight, mSurfaceFormat))
        return false;

    mState = cstStopped;
    emit stateChanged(cstStopped);
    return true;
}

bool PGMCamera::start()
{
    mState = cstStreaming;
    emit stateChanged(cstStreaming);
    QTimer::singleShot(0, this, [this](){startStreaming();});
    return true;
}

bool PGMCamera::stop()
{
    mState = cstStopped;
    emit stateChanged(cstStopped);

    return true;
}

void PGMCamera::close()
{
    stop();
    mState = cstClosed;
    emit stateChanged(cstClosed);
}

void PGMCamera::startStreaming()
{
    if(mState != cstStreaming)
        return;

    if(!mInputImage.data)
        return;

    while(mState == cstStreaming)
    {
        cudaMemcpy(mInputBuffer.getBuffer(), mInputImage.data.get(), mInputImage.wPitch * mInputImage.h, cudaMemcpyHostToDevice);
        mInputBuffer.release();
        QThread::msleep(1000 / mFPS);

        {
            QMutexLocker l(&mLock);
            mRawProc->wake();
        }

    }

}
bool PGMCamera::getParameter(cmrCameraParameter param, float& val)
{
    if(param < 0 || param > prmLast)
        return false;

    switch (param)
    {
    case prmFrameRate:
        val = mFPS;
        return true;

    case prmExposureTime:
        val = 1000 / mFPS;
        return true;

    default:
        break;
    }

    return false;
}

bool PGMCamera::setParameter(cmrCameraParameter param, float val)
{
    Q_UNUSED(param)
    Q_UNUSED(val)
    return false;
}

bool PGMCamera::getParameterInfo(cmrParameterInfo& info)
{
    Q_UNUSED(info)
    return false;
}
