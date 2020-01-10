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

#ifndef ASYNCJPEGWRITER_H
#define ASYNCJPEGWRITER_H

#include <QObject>
#include <QWaitCondition>
#include <QMutex>
#include <QThread>

#include "AsyncQueue.h"
#include "FastAllocator.h"
#include "MJPEGEncoder.h"
#include <memory>


struct FileWriterTask
{
    unsigned char* data;
    unsigned int size{};
    QString fileName;
};

class AsyncWriter : public QObject
{
    Q_OBJECT
public:
    explicit AsyncWriter(int size = -1, QObject *parent = nullptr);
    ~AsyncWriter();

    void initBuffers(unsigned bufferSize);
    unsigned char* getBuffer();
    void start();
    void stop();
    void put(FileWriterTask* task);
    void clear();
    void wake(){mStart.wakeAll();}
    void waitFinish();
    void setMaxSize(int sz);
    int  queueSize(){return mTasks.count();}
    int  getProcessedFrames(){return mProcessed;}
    int  getDroppedFrames(){return mDropped;}
    unsigned bufferSize() {return mBufferSize;}

signals:
    void progress(int percent);

public slots:

protected:
    virtual void processTask(FileWriterTask* task) = 0;

    void startWriting();

    bool mCancel {false};
    bool mWriting {false};

    unsigned mBufferSize {0};
    unsigned mCurrentBuffer{0};
    std::vector<std::unique_ptr<unsigned char, FastAllocator>> mBuffers;

    QMutex mLock;
    QWaitCondition mStart;
    QWaitCondition mFinish;
    QThread mWorkThread;
    AsyncQueue<FileWriterTask*> mTasks;

    int mMaxSize = -1;
    int mProcessed = 0;
    int mDropped = 0;

    const uint maxQueuSize = 32;
};


class AsyncFileWriter : public AsyncWriter
{
    Q_OBJECT
public:
    explicit AsyncFileWriter(int size = -1, QObject *parent = nullptr);

protected:
    virtual void processTask(FileWriterTask* task);
};


class AsyncMJPEGWriter : public AsyncWriter
{
    Q_OBJECT
public:
    explicit AsyncMJPEGWriter(int size = -1, QObject *parent = nullptr);
    bool open(int width, int height, int fps, fastJpegFormat_t fmt, const QString& outFileName);
    void close();

protected:
    virtual void processTask(FileWriterTask* task);

private:
    QScopedPointer<MJPEGEncoder> mEncoderPtr;
};
#endif // ASYNCJPEGWRITER_H
