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

#ifndef XIMEACAMERA_H
#define XIMEACAMERA_H

#ifdef SUPPORT_XIMEA

#include "GPUCameraBase.h"
#include <xiApi.h>

#include <FastAllocator.h>

class XimeaCamera : public GPUCameraBase
{
    Q_OBJECT
public:
    XimeaCamera();
    ~XimeaCamera();
    virtual bool open(uint32_t devID) override;
    virtual bool start() override;
    virtual bool stop() override;
    virtual void close() override;

    bool getParameter(cmrCameraParameter param, float& val) override;
    bool setParameter(cmrCameraParameter param, float val) override;
    bool getParameterInfo(cmrParameterInfo& info) override;
    GPUImage_t *getLastFrame() override;
protected:

private:
    HANDLE hDevice = 0;
    bool mStreaming = false;
    void startStreaming();

    bool mPacked = false;

    struct FastMemory{
        std::unique_ptr<unsigned char, FastAllocator> _data;
        unsigned _size{};

        void resize(unsigned newsize){
            if(_size == newsize){
                return;
            }
            if(newsize == 0){
                release();
                return;
            }
            _size = newsize;
            FastAllocator alloc;
            _data.reset((unsigned char*)alloc.allocate(_size));
        }
        void release(){
            _data.reset();
            _size = 0;
        }
        unsigned size() const {
            return _size;
        }
        unsigned char *data(){
            return _data.get();
        }
    };

    FastMemory frameData;
    QByteArray packedData;
};

#endif // SUPPORT_XIMEA

#endif // XIMEACAMERA_H
