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
#include "GPUCameraBase.h"
#include "MainWindow.h"
#include "FPNReader.h"
#include "FFCReader.h"

#include "avfilewriter/avfilewriter.h"

#include <QElapsedTimer>
#include <QDateTime>
#include <QDebug>
#include <QPoint>

RawProcessor::RawProcessor(GPUCameraBase *camera, GLRenderer *renderer):QObject(nullptr),
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
    QElapsedTimer tm;
    tm.start();
    while(mWorking && tm.elapsed() <= 1000)
    {
        QThread::msleep(100);
    }
}

void RawProcessor::wake()
{
    mWake = true;
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

    qint64 lastTime = 0;
    QElapsedTimer tm;
    tm.start();

    QByteArray buffer;
    buffer.resize(mOptions.Width * mOptions.Height * 4);

    int bpc = GetBitsPerChannelFromSurface(mCamera->surfaceFormat());
    int maxVal = (1 << bpc) - 1;
    QString pgmHeader = QString("P5\n%1 %2\n%3\n").arg(mOptions.Width).arg(mOptions.Height).arg(maxVal);

    mWake = false;

    while(mWorking)
    {
        if(!mWake)
        {
            mWaitMutex.lock();
            mWaitCond.wait(&mWaitMutex);
            mWaitMutex.unlock();
        }
        mWake = false;
        if(!mWorking)
            break;

        if(!mProcessorPtr || mCamera == nullptr)
            continue;

        GPUImage_t* img = mCamera->getFrameBuffer()->getLastImage();
        mProcessorPtr->Transform(img, mOptions);
        if(mRenderer)
        {
            qint64 curTime = tm.elapsed();
/// arm processor cannot show 60 fps
#ifdef __ARM_ARCH
            const qint64 frameTime = 32;
            if(curTime - lastTime >= frameTime)
#endif
            {
                if(mOptions.ShowPicture){
                    mRenderer->loadImage(mProcessorPtr->GetFrameBuffer(), mOptions.Width, mOptions.Height);
                    mRenderer->update();
                }
                lastTime = curTime;

                emit finished();
            }
        }

        /// added sending by rtsp
        if(mOptions.Codec == CUDAProcessorOptions::vcJPG ||
           mOptions.Codec == CUDAProcessorOptions::vcMJPG)
        {
            if(mRtspServer && mRtspServer->isConnected())
            {
                mRtspServer->addFrame(nullptr);
            }
        }
        if(mOptions.Codec == CUDAProcessorOptions::vcH264 || mOptions.Codec == CUDAProcessorOptions::vcHEVC)
        {
            if(mRtspServer && mRtspServer->isConnected())
            {
                unsigned char* data = (uchar*)buffer.data();
                mProcessorPtr->export8bitData((void*)data, true);

                mRtspServer->addFrame(data);
            }
        }

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
                unsigned char* buf = mFileWriterPtr->getBuffer();
                if(buf != nullptr)
                {
                    unsigned w = 0;
                    unsigned h = 0;
                    unsigned pitch = 0;
                    mProcessorPtr->exportRawData(nullptr, w, h, pitch);

                    int sz = pgmHeader.size() + pitch * h;

                    FileWriterTask* task = new FileWriterTask();
                    task->fileName =  QStringLiteral("%1/%2%3.pgm").arg(mOutputPath,mFilePrefix).arg(mFrameCnt);
                    task->size = sz;

                    task->data = buf;
                    memcpy(task->data, pgmHeader.toStdString().c_str(), pgmHeader.size());
                    unsigned char* data = task->data + pgmHeader.size();
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

            }else if(mOptions.Codec == CUDAProcessorOptions::vcH264 || mOptions.Codec == CUDAProcessorOptions::vcHEVC)
            {
                //unsigned char* buf = mFileWriterPtr->getBuffer();
                /*if(buf != nullptr)*/{
//                    unsigned char* data = (uchar*)buffer.data();
//                    mProcessorPtr->export8bitData((void*)data, true);

//                    int w = mOptions.Width;
//                    int h = mOptions.Height;
//                    int pitch = w * (mProcessorPtr->isGrayscale()? 1 : 3);
//                    int sz = pitch * h;

                    FileWriterTask* task = new FileWriterTask();
                    task->fileName =  QStringLiteral("%1/%2%3.mkv").arg(mOutputPath,mFilePrefix).arg(mFrameCnt);
                    task->size = 0;

                    task->data = nullptr;
                    //memcpy(task->data, data, sz);

                    mFileWriterPtr->put(task);
                    mFileWriterPtr->wake();

                    //mRtspServer->addFrame(data);
                }
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
            // to minimize delay in main thread
            mProcessorPtr->mut2.lock();
            ret = mProcessorPtr->stats2;
            mProcessorPtr->mut2.unlock();
        }

        if(mWriting)
        {
            ret[QStringLiteral("procFrames")] = mFileWriterPtr->getProcessedFrames();
            ret[QStringLiteral("droppedFrames")] = mFileWriterPtr->getDroppedFrames();
            AVFileWriter *obj = dynamic_cast<AVFileWriter*>(mFileWriterPtr.data());
            if(obj)
                ret[QStringLiteral("encoding")] = obj->duration();
        }
        else
        {
            ret[QStringLiteral("procFrames")] = -1;
            ret[QStringLiteral("droppedFrames")] = -1;
        }
        ret[QStringLiteral("acqTime")] = acqTimeNsec;

        if(mRtspServer){
            ret[QStringLiteral("encoding")] = mRtspServer->duration();
        }
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
                     mCamera->isColor() ? mOptions.JpegSamplingFmt : FAST_JPEG_Y,
                     fileName);
        mFileWriterPtr.reset(writer);
    }
    else if(mCodec == CUDAProcessorOptions::vcH264 || mCodec == CUDAProcessorOptions::vcHEVC){
        QString fileName = QDir::toNativeSeparators(
                    QStringLiteral("%1/%2.avi").
                    arg(mOutputPath).
                    arg(QDateTime::currentDateTime().toString(QStringLiteral("dd_MM_yyyy_hh_mm_ss"))));

        AVFileWriter *writer = new AVFileWriter();

        auto funEncodeNv12 = [this](unsigned char* yuv, int ){
            //int channels = dynamic_cast<CUDAProcessorGray*>(mProcessorPtr.data()) == nullptr? 3 : 1;

            mProcessorPtr->exportNV12DataDevice(yuv);
        };
        writer->setEncodeNv12Fun(funEncodeNv12);

#ifdef __ARM_ARCH
        auto funEncodeYuv = [this](unsigned char* yuv, int bitdepth){
            //int channels = dynamic_cast<CUDAProcessorGray*>(mProcessorPtr.data()) == nullptr? 3 : 1;

            if(bitdepth == 8)
                mProcessorPtr->exportYuv8Data(yuv);
            else
                mProcessorPtr->exportP010Data(yuv);
        };
        writer->setEncodeYUV420Fun(funEncodeYuv);
#else
        auto funEncodeP010 = [this](unsigned char* yuv, int){
            //int channels = dynamic_cast<CUDAProcessorGray*>(mProcessorPtr.data()) == nullptr? 3 : 1;

            mProcessorPtr->exportP010DataDevice(yuv);
        };
        writer->setEncodeYUV420Fun(funEncodeP010);
#endif

        writer->open(mCamera->width(),
                     mCamera->height(),
                     mOptions.bitrate,
                     60,
                     mCodec == CUDAProcessorOptions::vcHEVC,
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
    if(mCodec == CUDAProcessorOptions::vcH264 || mCodec == CUDAProcessorOptions::vcHEVC){
        AVFileWriter *writer = static_cast<AVFileWriter*>(mFileWriterPtr.data());
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

    if(!mCamera && !mCamera->isColor())
        return retClr;

    //qDebug() << rawPoint;

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

void RawProcessor::setRtspServer(const QString &url)
{
    if(url.isEmpty())
        return;

    if(mOptions.Width == 0 || mOptions.Height == 0){
        return;
    }
    mUrl = url;

    RTSPStreamerServer::EncoderType encType = RTSPStreamerServer::etJPEG;
    if(mOptions.Codec == CUDAProcessorOptions::vcH264)
        encType = RTSPStreamerServer::etNVENC;
    if(mOptions.Codec == CUDAProcessorOptions::vcHEVC)
        encType = RTSPStreamerServer::etNVENC_HEVC;

    mRtspServer.reset(new RTSPStreamerServer(mOptions.Width, mOptions.Height, 3, url, encType, mOptions.bitrate));

    mRtspServer->setMultithreading(false);

	auto funEncode = [this](int, unsigned char* , int width, int height, int, Buffer& output){

		int channels = dynamic_cast<CUDAProcessorGray*>(mProcessorPtr.data()) == nullptr? 3 : 1;

		unsigned pitch = channels *(((width + FAST_ALIGNMENT - 1) / FAST_ALIGNMENT ) * FAST_ALIGNMENT);
        unsigned sz = pitch * height;

		output.buffer.resize(sz);
		mProcessorPtr->exportJPEGData(output.buffer.data(), mOptions.JpegQuality, sz);
		output.size = sz;
    };
    mRtspServer->setUseCustomEncodeJpeg(true);
    mRtspServer->setEncodeFun(funEncode);

    auto funEncodeNv12 = [this](unsigned char* yuv, int ){
        mProcessorPtr->exportNV12DataDevice(yuv);
    };
    mRtspServer->setEncodeNv12Fun(funEncodeNv12);

#ifdef __ARM_ARCH
        auto funEncodeYuv = [this](unsigned char* yuv, int bitdepth){
            //int channels = dynamic_cast<CUDAProcessorGray*>(mProcessorPtr.data()) == nullptr? 3 : 1;

            if(bitdepth == 8)
                mProcessorPtr->exportYuv8Data(yuv);
            else
                mProcessorPtr->exportP010Data(yuv);
        };
        mRtspServer->setEncodeYUV420Fun(funEncodeYuv);
#else
        auto funEncodeP010 = [this](unsigned char* yuv, int){
            //int channels = dynamic_cast<CUDAProcessorGray*>(mProcessorPtr.data()) == nullptr? 3 : 1;

            mProcessorPtr->exportP010DataDevice(yuv);
        };
        mRtspServer->setEncodeYUV420Fun(funEncodeP010);
#endif

    mRtspServer->startServer();
}

QString RawProcessor::url() const
{
    return mUrl;
}

void RawProcessor::startRtspServer()
{
    if(mUrl.isEmpty())
        return;

    if(mCamera == nullptr)
        return;

    mCodec = mOptions.Codec;

    setRtspServer(mUrl);
}

void RawProcessor::stopRtspServer()
{
    mRtspServer.reset();
}

bool RawProcessor::isStartedRtsp() const
{
    return mRtspServer && mRtspServer->isStarted();
}

bool RawProcessor::isConnectedRtspClient() const
{
    return mRtspServer && mRtspServer->isConnected();
}
