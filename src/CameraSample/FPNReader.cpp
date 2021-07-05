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

#include "FPNReader.h"
#include "ppm.h"

#include <QVector>
#include <QElapsedTimer>
#include <MallocAllocator.h>

void FPNTIFFHandler(const char* module, const char* fmt, const va_list ap)
{
    // ignore errors and warnings (or handle them your own way)
    Q_UNUSED(module)
    Q_UNUSED(fmt)
    Q_UNUSED(ap)
}

FPNReader::FPNReader(const QString &fileName) :
    mWidth(0),
    mHeight(0),
    mPitch(0),
    mFileName(fileName)
{
    QString suf = QFileInfo(fileName).suffix().toLower();
    if(suf == QStringLiteral("pgm"))
    {
        readPGM(fileName);
    }
}

void FPNReader::readPGM(const QString& fileName)
{
    MallocAllocator alloc;
    unsigned char* bits = nullptr;

    uint width = 0;
    uint height = 0;
    uint pitch = 0;
    uint bitsPerPixel = 0;
    uint samples = 0;
    if(1 != loadPPM(fileName.toLocal8Bit(),
                    reinterpret_cast<void**>(&bits),
                    &alloc,
                    width, pitch, height,
                    bitsPerPixel, samples))
        return;

    if(samples != 1)
        return;

    mPitch = pitch;
    mHeight = height;
    mWidth = width;
    mBpp = bitsPerPixel;

    FastAllocator a;

    try
    {
        mFPNBuffer.reset(static_cast<unsigned char*>(a.allocate(mPitch * mHeight)));
    }
    catch(...)
    {
        alloc.deallocate(bits);
        return;
    }

    unsigned char* dst = mFPNBuffer.get();

    memcpy(dst, bits, mPitch * mHeight);
    alloc.deallocate(bits);
}


FPNStore* FPNStore::Instance()
{
    static FPNStore instance_;
    return &instance_;
}


FPNStore::~FPNStore()
{
    clear();
}

void FPNStore::clear()
{
    for(auto &p : fpnCache)
        delete p;
    fpnCache.clear();
}

void* FPNStore::getFPN(const QString &filename)
{
    FPNReader* reader = getReader(filename);
    if(reader)
        return reader->data();

    return nullptr;
}

FPNReader* FPNStore::getReader(const QString& filename)
{
    if(!QFileInfo::exists(filename))
    {
        return nullptr;
    }

    if(fpnCache.contains(filename))
        return fpnCache[filename];

    // Add fpn from file (if exists)
    auto* reader = new FPNReader(filename);
    if(reader->isValid())
    {
        fpnCache[filename] = reader;
        return reader;
    }
    delete reader;
    return nullptr;
}
