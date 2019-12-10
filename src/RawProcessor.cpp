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

#include "RawProcessor.h"
#include "CUDAProcessorBase.h"
#include "CUDAProcessorGray.h"
#include "FrameBuffer.h"
#include "CameraBase.h"
#include "MainWindow.h"
#include "FPNReader.h"
#include "FFCReader.h"

#include <QElapsedTimer>

RawProcessor::RawProcessor(CameraBase *camera, GLRenderer *renderer) :
    mCamera(camera),
    mRenderer(renderer),
    QObject(nullptr)
{


    mProcessorPtr.reset(new CUDAProcessorBase());

    connect(mProcessorPtr.data(), SIGNAL(error()), this, SIGNAL(error()));

    mCUDAThread.setObjectName(QStringLiteral("CUDAThread"));
    moveToThread(&mCUDAThread);
    mCUDAThread.start();
}

RawProcessor::~RawProcessor()
{
    stop();
    mCUDAThread.quit();
    mCUDAThread.wait(3000);
}

fastStatus_t RawProcessor::init()
{
    if(!mProcessorPtr)
        return FAST_INVALID_VALUE;

    unsigned pitch = 3 *(((mOptions.Width + FAST_ALIGNMENT - 1) / FAST_ALIGNMENT ) * FAST_ALIGNMENT);
    unsigned sz = pitch * mOptions.Height;
    mFileWriter.initBuffers(sz);

    return mProcessorPtr->Init(mOptions);
}

void RawProcessor::start()
{
    if(!mProcessorPtr || mCamera == nullptr)
        return;

    QTimer::singleShot(0, this, [this](){startWorking();});
}

void RawProcessor::stop()
{
    mWorking = false;
    mWaitCond.wakeAll();

    mFileWriter.waitFinish();
    mFileWriter.stop();

    //Wait up to 1 sec until mWorking == false
    QTime tm;
    tm.start();
    while(mWorking && tm.elapsed() <= 1000)
    {
        QThread::msleep(100);
    }
}

void RawProcessor::wake()
{
    mWaitCond.wakeAll();
}

void RawProcessor::updateOptions(const CUDAProcessorOptions& opts)
{
    if(!mProcessorPtr)
        return;
    QMutexLocker lock(&(mProcessorPtr->mut));
    mOptions = opts;
}

void RawProcessor::startWorking()
{
    mWorking = true;

    int frameTime = 1000 / mRenderFps;
    qint64 lastTime = 0;
    QElapsedTimer tm;
    tm.start();

    while(mWorking)
    {
        mWaitMutex.lock();
        mWaitCond.wait(&mWaitMutex);
        mWaitMutex.unlock();

        if(!mWorking)
            break;

        if(!mProcessorPtr || mCamera == nullptr)
            continue;

        ImageT* img = mCamera->getFrameBuffer()->getLastImage();
        mProcessorPtr->Transform(img, mOptions);

        if(mWriting)
        {
            if(mOptions.Codec == CUDAProcessorOptions::vcJPG)
            {
                unsigned char* buf = mFileWriter.getBuffer();
                if(buf != nullptr)
                {
                    FileWriterTask* task = new FileWriterTask();
                    task->fileName =  QStringLiteral("%1/%2%3.jpg").arg(mOutputPath).
                            arg(mFilePrefix).arg(mFrameCnt);
                    task->size = mFileWriter.bufferSize();
                    task->data = buf;
                    mProcessorPtr->exportJPEGData(task->data, mOptions.JpegQuality, task->size);
                    mFileWriter.put(task);
                    mFileWriter.wake();
                    mFrameCnt++;
                }
            }
            else if(mOptions.Codec == CUDAProcessorOptions::vcPGM)
            {
                int bpc = GetBitsPerChannelFromSurface(img->surfaceFmt);
                int maxVal = (1 << bpc) - 1;
                unsigned char* buf = mFileWriter.getBuffer();
                if(buf != nullptr)
                {
                    unsigned w = 0;
                    unsigned h = 0;
                    unsigned pitch = 0;
                    mProcessorPtr->exportRawData(nullptr, w, h, pitch);

                    QString header = QString("P5\n%1 %2\n%3\n").arg(w).arg(h).arg(maxVal);

                    int sz = header.size() + pitch * h;

                    FileWriterTask* task = new FileWriterTask();
                    task->fileName =  QStringLiteral("%1/%2%3.pgm").arg(mOutputPath).
                            arg(mFilePrefix).arg(mFrameCnt);
                    task->size = sz;

                    task->data = buf;
                    memcpy(task->data, header.toStdString().c_str(), header.size());
                    unsigned char* data = task->data + header.size();
                    mProcessorPtr->exportRawData((void*)data, w, h, pitch);

                    //Not 8 bit pgm requires big endian byte order
                    if(img->surfaceFmt != FAST_I8)
                    {
                        unsigned short* data16 = (unsigned short*)data;
                        for(unsigned i = 0; i < w * h; i++)
                        {
                            unsigned short val = *data16;
                            *data16 = (val << 8) | (val >> 8);
                            data16++;
                        }
                    }

                    mFileWriter.put(task);
                    mFileWriter.wake();
                    mFrameCnt++;
                }

            }
        }

        if(mRenderer)
        {
            qint64 curTime = tm.elapsed();
            if(curTime - lastTime >= frameTime)
            {
                mRenderer->loadImage(mProcessorPtr->GetFrameBuffer(), mOptions.Width, mOptions.Height);
                mRenderer->update();
                lastTime = curTime;

                emit finished();
            }
        }
    }
    mWorking = false;
}

fastStatus_t RawProcessor::getLastError()
{
    if(mProcessorPtr)
        return mProcessorPtr->getLastError();
    else
        return FAST_OK;
}

QString RawProcessor::getLastErrorDescription()
{
    if(mProcessorPtr)
        return mProcessorPtr->getLastErrorDescription();
    else
        return QString();
}

QMap<QString, float> RawProcessor::getStats()
{
    QMap<QString, float> ret;
    if(mProcessorPtr)
    {
        {
            QMutexLocker l(&mProcessorPtr->mut);
            ret = mProcessorPtr->stats;
        }

        if(mWriting)
        {
            ret[QStringLiteral("procFrames")] = mFileWriter.getProcessedFrames();
            ret[QStringLiteral("droppedFrames")] = mFileWriter.getDroppedFrames();
        }
        else
        {
            ret[QStringLiteral("procFrames")] = -1;
            ret[QStringLiteral("droppedFrames")] = -1;
        }
        ret[QStringLiteral("acqTime")] = acqTimeNsec;
    }

    return ret;
}

void RawProcessor::startWriting()
{
    mWriting = false;
    if(QFileInfo(mOutputPath).exists())
    {
        QDir dir;
        if(!dir.mkpath(mOutputPath))
            return;
    }

    if(!QFileInfo(mOutputPath).isDir())
        return;

    mFrameCnt = 0;
    mWriting = true;
}

void RawProcessor::setSAM(const QString& fpnFileName, const QString& ffcFileName)
{
    FPNReader* fpnReader = gFPNStore->getReader(fpnFileName);

    if(fpnReader)
    {
        auto bpp = GetBitsPerChannelFromSurface(mOptions.SurfaceFmt);
        if(fpnReader->width() != mOptions.Width ||
           fpnReader->height() != mOptions.Height ||
           fpnReader->bpp() != bpp)
        {
            mOptions.MatrixB = nullptr;
        }
        else
        {
            mOptions.MatrixB = fpnReader->data();
        }
    }
    else
        mOptions.MatrixB = nullptr;


    FFCReader* ffcReader = gFFCStore->getReader(ffcFileName);
    if(ffcReader)
    {
        if(ffcReader->width() != mOptions.Width ||
           ffcReader->height() != mOptions.Height)
        {
            mOptions.MatrixA = nullptr;
        }
        else
        {
            mOptions.MatrixA = ffcReader->data();
        }
    }
    else
        mOptions.MatrixA = nullptr;

    init();
}
