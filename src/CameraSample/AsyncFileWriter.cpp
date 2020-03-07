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

#include "AsyncFileWriter.h"
#include "MJPEGEncoder.h"

#include <QTimer>
#include <QFile>
#include <QFileInfo>
#include <QDir>

AsyncWriter::AsyncWriter(int size, QObject *parent):
    QObject(parent),
    mMaxSize(size)
{
    mWorkThread.setObjectName(QStringLiteral("File Writer Thread"));
    moveToThread(&mWorkThread);

    mWorkThread.start();
    start();
}

AsyncWriter::~AsyncWriter()
{
    stop();
    clear();
    mWorkThread.quit();
    mWorkThread.wait(3000);
}

void AsyncWriter::start()
{
    QTimer::singleShot(0, this, [this](){
        if(!this->mWriting) this->startWriting();
    });
}

void AsyncWriter::initBuffers(unsigned bufferSize)
{
    if(bufferSize <= mBufferSize)
        return;

    if(mBuffers.empty())
        mBuffers.resize(maxQueuSize);
    FastAllocator alloc;
    for(auto & buffer : mBuffers )
        buffer.reset((unsigned char*)alloc.allocate(bufferSize));
    mBufferSize = bufferSize;
}

unsigned char* AsyncWriter::getBuffer()
{
    if(mBuffers.empty())
        return nullptr;

    unsigned char* ret = mBuffers[mCurrentBuffer].get();

    mCurrentBuffer = (mCurrentBuffer + 1) % maxQueuSize;

    return ret;
}

void AsyncWriter::put(FileWriterTask* task)
{
    if(mTasks.count() > maxQueuSize)
    {
        delete task;
        mDropped++;
    }
    else
        mTasks.push(task);
}

void AsyncWriter::setMaxSize(int sz)
{
    if(sz <= 0)
        return;
    mMaxSize = sz;
    mProcessed = 0;
}

void AsyncWriter::startWriting()
{
    mWriting = true;
    QMutex mut;

    mProcessed = 0;
    mDropped = 0;

    while(!mCancel)
    {
        mut.lock();
        mStart.wait(&mut);
        mut.unlock();

        while(mTasks.count() > 0)
        {
            FileWriterTask* task = mTasks.pop();
            if(task)
            {
                processTask(task);
                mProcessed++;
                if(mMaxSize >= 0)
                {
                    if(mProcessed <= mMaxSize)
                        emit progress((mProcessed * 100) / mMaxSize);
                }
                delete task;
            }
        }
        mFinish.wakeAll();
    }
    mWriting = false;
}

void AsyncWriter::waitFinish()
{
    if(mTasks.isEmpty())
        return;

    QMutex lock;
    QMutexLocker l(&lock);
    mFinish.wait(&lock, 3000);
}


void AsyncWriter::stop()
{
    mCancel = true;
    wake();
}

void AsyncWriter::clear()
{
    QMutexLocker lock(&mLock);
    while(mTasks.count() > 0)
        delete mTasks.pop();
}

AsyncFileWriter::AsyncFileWriter(int size, QObject *parent):
    AsyncWriter(size, parent)
{
    mMaxSize = size;
    mWorkThread.setObjectName(QStringLiteral("File Writer Thread"));
    moveToThread(&mWorkThread);

    mWorkThread.start();
    start();
}

void AsyncFileWriter::processTask(FileWriterTask* task)
{
    if(task == nullptr)
        return;

    QString path = QFileInfo(task->fileName).path();
    QDir dir(path);
    if(!dir.exists())
    {
        if(!dir.mkpath(path))
            return;
    }

    QFile f(task->fileName);
    if(f.open(QFile::WriteOnly))
    {
        f.write((char*)task->data, task->size);
        f.close();
    }
}


AsyncMJPEGWriter::AsyncMJPEGWriter(int size, QObject *parent):
    AsyncWriter(size, parent)
{
    mMaxSize = size;
    mWorkThread.setObjectName(QStringLiteral("MJPEG Writer Thread"));
    moveToThread(&mWorkThread);

    mWorkThread.start();
    start();
}

bool AsyncMJPEGWriter::open(int width, int height, int fps, fastJpegFormat_t fmt, const QString& outFileName)
{
    if(!QFileInfo::exists(QFileInfo(outFileName).path()))
        return false;

    if(mEncoderPtr)
        mEncoderPtr->close();

    mEncoderPtr.reset(new MJPEGEncoder(width, height, fps, fmt, outFileName));
    return mEncoderPtr->isOpened();

}

void AsyncMJPEGWriter::close()
{
    if(!mEncoderPtr)
        return;

    mEncoderPtr->close();
}

void AsyncMJPEGWriter::processTask(FileWriterTask* task)
{
    if(task == nullptr)
        return;

    if(!mEncoderPtr)
        return;

    mEncoderPtr->addJPEGFrame(task->data, int(task->size));
}
