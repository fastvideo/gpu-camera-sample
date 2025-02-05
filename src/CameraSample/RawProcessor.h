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

#ifndef RAWPROCESSOR_H
#define RAWPROCESSOR_H

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QScopedPointer>
#include <QDir>
#include <QColor>

#include "CUDAProcessorOptions.h"
#include "AsyncFileWriter.h"
#include "RTSPStreamerServer.h"

class CUDAProcessorBase;
class CircularBuffer;
class MainWindow;
class GLRenderer;
class GPUCameraBase;

class RawProcessor : public QObject
{
    Q_OBJECT
public:
    explicit RawProcessor(QSharedPointer<GPUCameraBase> camera, QSharedPointer<GLRenderer> renderer);
    ~RawProcessor();

    fastStatus_t init();
    void start();
    void stop();
    void wake();
    void updateOptions(const CUDAProcessorOptions& opts);
    CUDAProcessorBase*   getCUDAProcessor() {return mProcessorPtr.data();}
    fastStatus_t         getLastError();
    QString              getLastErrorDescription();
    QMap<QString, float> getStats();
    void startWriting();
    void stopWriting();
    void setOutputPath(const QString& path){mOutputPath = path;}
    void setFilePrefix(const QString& prefix){mFilePrefix = prefix;}
    void setSAM(const QString& fpnFileName, const QString& ffcFileName);

    QColor getAvgRawColor(QPoint rawPoint);

    void setRtspServer(const QString& url);
    QString url() const;
    void startRtspServer();
    void stopRtspServer();
    bool isStartedRtsp() const;
    bool isConnectedRtspClient() const;

    float acqTimeNsec = -1.;

signals:
    void finished();
    void error();

public slots:

private:
    std::atomic_bool     mWorking = false;
    bool                 mWriting = false;
    CUDAProcessorOptions mOptions;
    CUDAProcessorOptions::VideoCodec mCodec = CUDAProcessorOptions::vcNone;

    QString mFPNFile;
    QString mFFCFile;

    QScopedPointer<CUDAProcessorBase> mProcessorPtr;
    QScopedPointer<AsyncWriter>       mFileWriterPtr;
    QMutex               mWaitMutex;
    QWaitCondition       mWaitCond;
    bool                 mWake = false;
    QSharedPointer<GPUCameraBase>  mCamera;
    QSharedPointer<GLRenderer>     mRenderer = nullptr;
    QThread              mCUDAThread;
    float                mRenderFps = 30;
    QString              mOutputPath;
    QString              mFilePrefix;
    unsigned             mFrameCnt = 0;
    QString              mUrl;
    QScopedPointer<RTSPStreamerServer> mRtspServer;


    void startWorking();
};

//class AsyncCUDATransformer : public QObject
//{
//    Q_OBJECT
//public:
//    explicit AsyncCUDATransformer(CUDAProcessorBase* proc);

//    void start();
//    void stop();
//    void wake();
//private:
//    CUDAProcessorBase* mProcessor = nullptr;
//    bool mWorking = false;
//};

#endif // RAWPROCESSOR_H
