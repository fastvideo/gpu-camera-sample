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

#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <vector>
#include <thread>
#include <functional>
#include <chrono>

const size_t MAX_WIDTH_RTP_JPEG = 2000;     /// by rfc 2435: 2040
const size_t MAX_HEIGHT_RTP_JPEG = 2000;    /// by rfc 2435: 2040

const size_t MAX_WIDTH_JPEG = 1024;
const size_t MAX_HEIGHT_JPEG = 1024;

#ifdef _MSC_VER
    typedef std::chrono::steady_clock::time_point time_point;
#else
    typedef std::chrono::system_clock::time_point time_point;
#endif

typedef std::vector <unsigned char > bytearray;

struct Buffer{
	bytearray buffer;
	size_t size = 0;
};

typedef std::function<void(int, unsigned char* data, int width, int height, int channels, Buffer& output)> TEncodeRgb;

typedef std::function<void(/* out */unsigned char *yuv,
                           int bitdepth)> TEncodeFun;

namespace rtp_packet_add_header
{
    const size_t sizeof_header = 4 + 4 + 4;              /// header + xOff + yOff + cntX + cntY + width + height
    const uint8_t HEADER[4] = {'S', 'G', 'M', 'T'};     /// segment -> SGMT

    union UShort{
        unsigned char uc[2];
        unsigned short us;
    };

    inline void setHeader(unsigned char *header, size_t xOff, size_t yOff, size_t cntX, size_t cntY, unsigned short width, unsigned short height){
        UShort w, h;
        w.us = width;
        h.us = height;
        header[12] = header[0];                         // last two bytes in original jpeg packet is
        header[13] = header[1];                         // marker of end. move to last place of new header
        header[0] = static_cast<unsigned char>(xOff);
        header[1] = static_cast<unsigned char>(yOff);
        header[2] = static_cast<unsigned char>(cntX);
        header[3] = static_cast<unsigned char>(cntY);
        header[4] = w.uc[0];
        header[5] = w.uc[1];
        header[6] = h.uc[0];
        header[7] = h.uc[1];

        header[8] = HEADER[0];
        header[9] = HEADER[1];
        header[10] = HEADER[2];
        header[11] = HEADER[3];
    }

    inline bool getHeader(unsigned char *header, size_t &xOff, size_t &yOff, size_t &cntX, size_t &cntY, unsigned short &width, unsigned short &height)
    {
        if(header[8] == HEADER[0] && header[9] == HEADER[1] && header[10] == HEADER[2] && header[11] == HEADER[3]){
            UShort w, h;
            xOff = static_cast<size_t>(header[0]);
            yOff = static_cast<size_t>(header[1]);
            cntX = static_cast<size_t>(header[2]);
            cntY = static_cast<size_t>(header[3]);
            w.uc[0] = header[4];
            w.uc[1] = header[5];
            h.uc[0] = header[6];
            h.uc[1] = header[7];
            width = w.us;
            height = h.us;
            header[0] = header[12];
            header[1] = header[13];
            return true;
        }
        return false;
    }
}

typedef std::unique_ptr< std::thread > pthread;

/// get duration of exectute function
template< typename F, class... Types >
double GETDUR(F fun, Types ...args)
{
  auto starttime = std::chrono::steady_clock::now();
  fun(args...);
  auto dur = std::chrono::steady_clock::now() - starttime;
  double tm = std::chrono::duration_cast<std::chrono::microseconds>(dur).count()/1000.;
  return tm;
}

typedef std::chrono::steady_clock::time_point timepoint;

inline timepoint getNow()
{
    return std::chrono::steady_clock::now();
}

inline double getDuration(timepoint start)
{
    auto dur = std::chrono::steady_clock::now() - start;
	double duration = std::chrono::duration_cast<std::chrono::microseconds>(dur).count()/1000.;
	return duration;
}

#endif // COMMON_H
