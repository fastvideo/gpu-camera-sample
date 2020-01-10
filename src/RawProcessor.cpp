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
#include <QDateTime>
#include <QDebug>
#include <QPoint>

RawProcessor::RawProcessor(CameraBase *camera, GLRenderer *renderer):QObject(nullptr),
    mCamera(camera),
    mRenderer(renderer)
{
    if(camera->isColor())
        mProcessorPtr.reset(new CUDAProcessorBase());
    else
        mProcessorPtr.reset(new CUDAProcessorGray());

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

    if(mFileWriterPtr)
    {
        mFileWriterPtr->waitFinish();
        mFileWriterPtr->stop();
    }

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

    int frameTime = qRound(1000.F / mRenderFps);
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

        if(mWriting && mFileWriterPtr)
        {
            if(mOptions.Codec == CUDAProcessorOptions::vcJPG ||
               mOptions.Codec == CUDAProcessorOptions::vcMJPG)
            {
                unsigned char* buf = mFileWriterPtr->getBuffer();
                if(buf != nullptr)
                {
                    FileWriterTask* task = new FileWriterTask();
                    task->fileName =  QStringLiteral("%1/%2%3.jpg").arg(mOutputPath,mFilePrefix).arg(mFrameCnt);
                    task->size = mFileWriterPtr->bufferSize();
                    task->data = buf;
                    mProcessorPtr->exportJPEGData(task->data, mOptions.JpegQuality, task->size);
                    mFileWriterPtr->put(task);
                    mFileWriterPtr->wake();
                    mFrameCnt++;
                }
            }
            else if(mOptions.Codec == CUDAProcessorOptions::vcPGM)
            {
                int bpc = GetBitsPerChannelFromSurface(img->surfaceFmt);
                int maxVal = (1 << bpc) - 1;
                unsigned char* buf = mFileWriterPtr->getBuffer();
                if(buf != nullptr)
                {
                    unsigned w = 0;
                    unsigned h = 0;
                    unsigned pitch = 0;
                    mProcessorPtr->exportRawData(nullptr, w, h, pitch);

                    QString header = QString("P5\n%1 %2\n%3\n").arg(w).arg(h).arg(maxVal);

                    int sz = header.size() + pitch * h;

                    FileWriterTask* task = new FileWriterTask();
                    task->fileName =  QStringLiteral("%1/%2%3.pgm").arg(mOutputPath,mFilePrefix).arg(mFrameCnt);
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

                    mFileWriterPtr->put(task);
                    mFileWriterPtr->wake();
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
    return  (mProcessorPtr) ? mProcessorPtr->getLastErrorDescription() : QString();
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
            ret[QStringLiteral("procFrames")] = mFileWriterPtr->getProcessedFrames();
            ret[QStringLiteral("droppedFrames")] = mFileWriterPtr->getDroppedFrames();
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
    if(mCamera == nullptr)
        return;

    mWriting = false;
    if(QFileInfo(mOutputPath).exists())
    {
        QDir dir;
        if(!dir.mkpath(mOutputPath))
            return;
    }

    if(!QFileInfo(mOutputPath).isDir())
        return;

    mCodec = mOptions.Codec;

    if(mCodec == CUDAProcessorOptions::vcMJPG)
    {
        QString fileName = QDir::toNativeSeparators(
                    QStringLiteral("%1/%2.avi").
                    arg(mOutputPath).
                    arg(QDateTime::currentDateTime().toString(QStringLiteral("dd_MM_yyyy_hh_mm_ss"))));
        AsyncMJPEGWriter* writer = new AsyncMJPEGWriter();
        writer->open(mCamera->width(),
                     mCamera->height(),
                     25,
                     mCamera->isColor() ? mOptions.JpegSamplingFmt : JPEG_Y,
                     fileName);
        mFileWriterPtr.reset(writer);
    }
    else
        mFileWriterPtr.reset(new AsyncFileWriter());

    unsigned pitch = 3 *(((mOptions.Width + FAST_ALIGNMENT - 1) / FAST_ALIGNMENT ) * FAST_ALIGNMENT);
    unsigned sz = pitch * mOptions.Height;
    mFileWriterPtr->initBuffers(sz);

    mFrameCnt = 0;
    mWriting = true;
}

void RawProcessor::stopWriting()
{
    mWriting = false;
    if(!mFileWriterPtr)
    {
        mCodec = CUDAProcessorOptions::vcNone;
        return;
    }

    if(mCodec == CUDAProcessorOptions::vcMJPG)
    {
        AsyncMJPEGWriter* writer = static_cast<AsyncMJPEGWriter*>(mFileWriterPtr.data());
        writer->close();
    }

    mCodec = CUDAProcessorOptions::vcNone;
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

QColor RawProcessor::getAvgRawColor(QPoint rawPoint)
{
    QColor retClr = QColor(Qt::white);

    if(!mProcessorPtr)
        return retClr;

    qDebug() << rawPoint;

    unsigned int w = 0;
    unsigned int h = 0;
    unsigned int pitch = 0;
    fastStatus_t ret = FAST_OK;

    {
        QMutexLocker locker(&(mProcessorPtr->mut));
        ret =  mProcessorPtr->exportLinearizedRaw(nullptr, w, h, pitch);
    }

    std::unique_ptr<unsigned char, FastAllocator> linearBits16;
    FastAllocator allocator;
    size_t sz = pitch * h * sizeof(unsigned short);

    try
    {
        linearBits16.reset(static_cast<unsigned char*>(allocator.allocate(sz)));
    }
    catch(...)
    {
        return retClr;
    }

    {
        QMutexLocker locker(&(mProcessorPtr->mut));
        ret =  mProcessorPtr->exportLinearizedRaw(linearBits16.get(), w, h, pitch);
    }
    if(ret != FAST_OK)
        return retClr;

    int pickerSize = 4;

    if(rawPoint.x() % 2 != 0)
        rawPoint.rx()--;

    if(rawPoint.x() < pickerSize)
        rawPoint.setX(pickerSize);
    if(rawPoint.x() >= int(w) - pickerSize)
        rawPoint.setX(int(w) - pickerSize);

    if(rawPoint.y() % 2 != 0)
        rawPoint.ry()--;
    if(rawPoint.y() < pickerSize)
        rawPoint.setY(pickerSize);
    if(rawPoint.y() >= int(h) - pickerSize)
        rawPoint.setY(int(h) - pickerSize);

    int rOffset = 0;
    int g1Offset = 0;
    int g2Offset = 0;
    int bOffset = 0;

    int rowWidth = pitch / sizeof(unsigned short);

    if(mOptions.BayerFormat == FAST_BAYER_RGGB)
    {
        rOffset = 0;
        g1Offset = 1;
        g2Offset = rowWidth;
        bOffset = rowWidth + 1;
    }
    else if(mOptions.BayerFormat == FAST_BAYER_BGGR)
    {
        bOffset = 0;
        g1Offset = 1;
        g2Offset = rowWidth;
        rOffset = rowWidth + 1;
    }
    else if(mOptions.BayerFormat == FAST_BAYER_GBRG)
    {
        g1Offset = 0;
        bOffset = 1;
        rOffset = rowWidth;
        g2Offset = rowWidth + 1;
    }
    else if(mOptions.BayerFormat == FAST_BAYER_GRBG)
    {
        g1Offset = 0;
        rOffset = 1;
        bOffset = rowWidth;
        g2Offset = rowWidth + 1;
    }
    else
        return {};

    int x = rawPoint.x();
    int y = rawPoint.y();
    auto * rawBits = reinterpret_cast<unsigned short*>(linearBits16.get());

    int r = 0;
    int g = 0;
    int b = 0;
    int cnt = 0;

    for(x = rawPoint.x() - pickerSize; x < rawPoint.x() + pickerSize; x += 2)
    {
        for(y = rawPoint.y() - pickerSize; y < rawPoint.y() + pickerSize; y += 2)
        {
            unsigned short* pixelPtr = rawBits + y * rowWidth + x;

            unsigned int val = pixelPtr[rOffset];
            r += val;

            val = pixelPtr[g1Offset] + pixelPtr[g2Offset];
            g += val;

            val = pixelPtr[bOffset];
            b += val;

            cnt++;
        }
    }

    if(cnt > 1)
    {
        r /= cnt;
        g /= 2 * cnt;
        b /= cnt;
    }

    return {qRgba64(quint16(r), quint16(g), quint16(b), 0)};

}
