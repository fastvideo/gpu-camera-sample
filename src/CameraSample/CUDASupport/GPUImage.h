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

#ifndef GPUIMAGE_H
#define GPUIMAGE_H

#include "fastvideo_sdk.h"
#include <string>
#include <memory>
#include <cstring>
#include "alignment.hpp"
#include "CudaAllocator.h"

template<class T>
class GPUImage {
public:
    std::unique_ptr<T, CudaAllocator> data;
    unsigned w;
    unsigned h;
    unsigned wPitch;
    unsigned bitsPerChannel;

    fastSurfaceFormat_t surfaceFmt;

    GPUImage(void) {
        w = h = wPitch = 0;
        bitsPerChannel = 8;
    };

    GPUImage(const GPUImage &img) {
        w = img.w;
        h = img.h;
        wPitch = img.wPitch;
        bitsPerChannel = img.bitsPerChannel;
        surfaceFmt = img.surfaceFmt;

        unsigned fullSize = wPitch * h;

        try
        {
			data.reset((T*)CudaAllocator::allocate(fullSize));
        }
        catch (std::bad_alloc& ba)
        {
            fprintf(stderr, "Memory allocation failed: %s\n", ba.what());
            return;
        }
        cudaMemcpy(data.get(), img.data.get(), fullSize * sizeof(T), cudaMemcpyDeviceToDevice);
    };

    unsigned GetBytesPerPixel() const {
        return uDivUp(bitsPerChannel, 8u);
    }
};
#endif // GPUIMAGE_H
