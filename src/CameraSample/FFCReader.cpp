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

#include "FFCReader.h"
//#include <tiffio.h>
#include "ppm.h"

#include <QVector>
#include <QString>
#include <QFileInfo>
#include <QElapsedTimer>
#include <MallocAllocator.h>

#ifndef WIN32
//#include <xmmintrin.h> //SSE Intrinsic Functions
#endif

#ifdef WIN32
template<unsigned i>
float vectorGetByIndex( __m128 V) {
    union {
        __m128 v;
        float a[4];
    } converter{};
    converter.v = V;
    return converter.a[i];
}
#endif

#ifdef WIN32
#include <intrin.h>
inline void ffcSortSSE(__m128& a, __m128& b)
{
    const __m128 t = a;
    a = _mm_min_ps(t, b);
    b = _mm_max_ps(t, b);
}

inline float ffcMedianSSE(__m128 a[9])
{
    ffcSortSSE(a[1], a[2]);
    ffcSortSSE(a[4], a[5]);
    ffcSortSSE(a[7], a[8]);
    ffcSortSSE(a[0], a[1]);
    ffcSortSSE(a[3], a[4]);
    ffcSortSSE(a[6], a[7]);
    ffcSortSSE(a[1], a[2]);
    ffcSortSSE(a[4], a[5]);
    ffcSortSSE(a[7], a[8]);
    ffcSortSSE(a[0], a[3]);
    ffcSortSSE(a[5], a[8]);
    ffcSortSSE(a[4], a[7]);
    ffcSortSSE(a[3], a[6]);
    ffcSortSSE(a[1], a[4]);
    ffcSortSSE(a[2], a[5]);
    ffcSortSSE(a[4], a[7]);
    ffcSortSSE(a[4], a[2]);
    ffcSortSSE(a[6], a[4]);
    ffcSortSSE(a[4], a[2]);

    return a[4].m128_f32[0];
}
#endif

void FFCTIFFHandler(const char* module, const char* fmt, va_list ap)
{
    // ignore errors and warnings (or handle them your own way)
    Q_UNUSED(module)
    Q_UNUSED(fmt)
    Q_UNUSED(ap)
}

FFCReader::FFCReader(const QString &fileName) :
    mWidth(0),
    mHeight(0),
    mPitch(0),
    mBoxH(32),
    mBoxW(32),
    mFileName(fileName)
{
    QString suf = QFileInfo(fileName).suffix().toLower();
    QVector<float> cfa;

    if(suf == QStringLiteral("pgm"))
    {
        readPGM(fileName, cfa);
    }
    if(cfa.isEmpty())
        return;

    FastAllocator alloc;
    try
    {
        mFFCBuffer.reset(static_cast<float*>(alloc.allocate(mPitch * mHeight)));
    }
    catch(...)
    {
        return;
    }
    QVector<float> cfaTmp(cfa.size());

    float refColor[2][2];

    cfaMedian(cfa.data(), cfaTmp.data());
    cfaBoxBlur(cfaTmp.data(), cfa.data());

    //find centre average values by channel
    for (int m = 0; m < 2; m++)
    {
        for (int n = 0; n < 2; n++)
        {
            int row = 2 * (mHeight >> 2) + m;
            int col = 2 * (mWidth >> 2) + n;
            refColor[m][n] = qMax(0.0f, cfa[row * mWidth + col]);
        }
    }
    float limitFactor = 1.f;
    for (int m = 0; m < 2; m++)
    {
        for (int n = 0; n < 2; n++)
        {
            refColor[m][n] *= limitFactor;
        }
    }

    // if the pixel value in the flat field is less or equal this value,
    // no correction will be applied.
    constexpr float minValue = 1.f;
    float* dst = mFFCBuffer.get();
    for (int row = 0; row < mHeight; row ++)
    {
        for (int col = 0; col < mWidth; col ++)
        {
            float blur = cfa[row * mWidth + col];
            dst[row * mWidth + col] = blur <= minValue ? 1.f : refColor[row & 1][col & 1] / blur;
        }
    }
}
void FFCReader::readPGM(const QString& fileName, QVector<float>& cfa)
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

    mPitch = width * sizeof(float);
    mHeight = height;
    mWidth = width;

    unsigned nPixels = width * height;
    cfa.resize(nPixels);

    if(bitsPerPixel == 8)
    {
        auto* src = static_cast<unsigned char*>(bits);
        for(uint i = 0; i < nPixels; i++)
        {
            cfa[i] = float(src[i]);
        }
    }
    else
    {
        auto* src = reinterpret_cast<unsigned short*>(bits);
        for(uint i = 0; i < nPixels; i++)
        {
            cfa[i] = float(src[i]);
        }
    }

    alloc.deallocate(bits);
}

FFCStore* FFCStore::Instance()
{
    static FFCStore instance_;
    return &instance_;
}


FFCStore::~FFCStore()
{
    clear();
}

void FFCStore::clear()
{
    for(auto &p : ffcCache)
        delete p;
    ffcCache.clear();
}

FFCReader* FFCStore::getReader(const QString& filename)
{
    if( !QFileInfo::exists(filename) )
        return nullptr;
    auto itr = ffcCache.find(filename);
    if(itr != ffcCache.end() )
        return itr.value();

    // Add ffc from file (if exists)
    auto * reader = new FFCReader(filename);
    if(reader->isValid())
    {
        ffcCache.insert(filename,reader);
        return reader;
    }
    delete reader;
    return nullptr;
}

float* FFCStore::getFFC(const QString& filename)
{
    FFCReader* reader = getReader(filename);
    if(reader)
        return reader->data();
    return nullptr;
}
void FFCReader::cfaMedian(float* src, float* dst)
{
#ifdef WIN32
    __m128 area[9];

    //Top 2 pixel margin
    memcpy(dst, src, 2 * mWidth * sizeof(float));

    //Bottom 2 pixel margin
    memcpy(dst + (mHeight - 2) * mWidth, src + (mHeight - 2) * mWidth,  2 * mWidth * sizeof(float));

    //Left and right 2 pixel margin
    for (int y = 2; y < mHeight - 2; y++)
    {
        dst[y * mWidth] = src[y * mWidth];
        dst[y * mWidth + 1] = src[y * mWidth + 1];
        dst[y * mWidth + mWidth - 2] = src[y * mWidth + mWidth - 2];
        dst[y * mWidth + mWidth - 1] = src[y * mWidth + mWidth - 1];
    }

    for(int y = 2; y < mHeight - 2; y++)
    {
        for(int x = 2; x < mWidth - 2; x++)
        {
            //1st row
            int pos = (y - 2) * mWidth + x - 2;//CFA(i - 2, j - 2)
            area[0] = _mm_set1_ps(src[pos]);

            pos += 2; //CFA(i - 2, j)
            area[1] = _mm_set1_ps(src[pos]);

            pos += 2; //CFA(i - 2, j + 2)
            area[2] = _mm_set1_ps(src[pos]);

            //2nd row
            pos += 2 * mWidth - 4; //CFA(i, j - 2)
            area[3] = _mm_set1_ps(src[pos]);

            pos += 2; //Current position
            area[4] = _mm_set1_ps(src[pos]);


            pos += 2; //CFA(i, j + 2)
            area[5] = _mm_set1_ps(src[pos]);

            //3rd row
            pos += 2 * mWidth - 4; //CFA(i + 2, j - 2)
            area[6] = _mm_set1_ps(src[pos]);

            pos += 2; //CFA(i + 2, j)
            area[7] = _mm_set1_ps(src[pos]);

            pos += 2; //CFA(i + 2, j + 2)
            area[8] = _mm_set1_ps(src[pos]);
            dst[x + y * mWidth] = ffcMedianSSE(area);
        }
    }
#endif
}

void FFCReader::cfaBoxBlur(float* src, float* dst)
{
    if (mBoxW < 0 || mBoxH < 0 || (mBoxW == 0 && mBoxH == 0))
    {
        // nothing to blur or negative values
        memcpy(dst, src, mWidth * mHeight * sizeof(float));
        return;
    }

    std::vector<float> tmpBuffer;
    float* cfatmp = nullptr;
    float* srcVertical = nullptr;


    if(mBoxH > 0 && mBoxW > 0)
    {
        // we need a temporary buffer if we have to blur both directions
        tmpBuffer.resize(mWidth * mHeight);
    }

    if(mBoxH == 0)
    {
        // if boxH == 0 we can skip the vertical blur and process the horizontal blur from riFlatFile to cfablur without using a temporary buffer
        cfatmp = dst;
    }
    else
        cfatmp = tmpBuffer.data();

    if(mBoxW == 0)
    {
        // if boxW == 0 we can skip the horizontal blur and process the vertical blur from riFlatFile to cfablur without using a temporary buffer
        srcVertical = src;
    }
    else
        srcVertical = cfatmp;

#pragma omp parallel
    {
        if(mBoxW > 0)
        {
            //box blur cfa image
            //horizontal blur
#pragma omp for

            for (int row = 0; row < mHeight; row++)
            {
                int len = mBoxW / 2 + 1;
                cfatmp[row * mWidth + 0] = src[row * mWidth + 0] / len;
                cfatmp[row * mWidth + 1] = src[row * mWidth + 1] / len;

                //Most left point point
                for (int j = 2; j <= mBoxW; j += 2)
                {
                    cfatmp[row * mWidth + 0] += src[row * mWidth + j] / len;
                    cfatmp[row * mWidth + 1] += src[row * mWidth + j + 1] / len;
                }
                //Left half-window margin
                for (int col = 2; col <= mBoxW; col += 2)
                {
                    cfatmp[row * mWidth + col] = (cfatmp[row * mWidth + col - 2] * len + src[row * mWidth + mBoxW + col]) / (len + 1);
                    cfatmp[row * mWidth + col + 1] = (cfatmp[row * mWidth + col - 1] * len + src[row * mWidth + mBoxW + col + 1]) / (len + 1);
                    len++;
                }
                //Body
                for (int col = mBoxW + 2; col < mWidth - mBoxW; col++)
                {
                    cfatmp[row * mWidth + col] = cfatmp[row * mWidth + col - 2] + (src[row * mWidth + mBoxW + col] - cfatmp[row * mWidth + col - mBoxW - 2]) / len;
                }

                //Right half-window margin
                for (int col = mWidth - mBoxW; col < mWidth; col += 2)
                {
                    cfatmp[row * mWidth + col] = (cfatmp[row * mWidth + col - 2] * len - cfatmp[row * mWidth + col - mBoxW - 2]) / (len - 1);

                    if (col + 1 < mWidth)
                    {
                        cfatmp[row * mWidth + col + 1] = (cfatmp[row * mWidth + col - 1] * len - cfatmp[row * mWidth + col - mBoxW - 1]) / (len - 1);
                    }
                    len --;
                }
            }
        }

        if(mBoxH > 0)
        {
            //vertical blur
#pragma omp for

            for (int col = 0; col < mWidth; col++)
            {
                int len = mBoxH / 2 + 1;
                dst[0 * mWidth + col] = srcVertical[0 * mWidth + col] / len;
                dst[1 * mWidth + col] = srcVertical[1 * mWidth + col] / len;


                //Most top point point
                for (int i = 2; i <= mBoxH; i += 2)
                {
                    dst[0 * mWidth + col] += srcVertical[i * mWidth + col] / len;
                    dst[1 * mWidth + col] += srcVertical[(i + 1) * mWidth + col] / len;
                }

                //Top half-window margin
                for (int row = 2; row <= mBoxH + 2; row += 2)
                {
                    dst[row * mWidth + col] = (dst[(row - 2) * mWidth + col] * len + srcVertical[(row + mBoxH) * mWidth + col]) / (len + 1);
                    dst[(row + 1) * mWidth + col] = (dst[(row - 1) * mWidth + col] * len + srcVertical[(row + mBoxH + 1) * mWidth + col]) / (len + 1);
                    len++;
                }

                //Body
                for (int row = mBoxH + 2; row < mHeight - mBoxH; row++)
                {
                    dst[row * mWidth + col] = dst[(row - 2) * mWidth + col] + (srcVertical[(row + mBoxH) * mWidth + col] - srcVertical[(row - mBoxH - 2) * mWidth + col]) / len;
                }

                //Bottom half-window margin
                for (int row = mHeight - mBoxH; row < mHeight; row += 2)
                {
                    dst[row * mWidth + col] = (dst[(row - 2) * mWidth + col] * len - srcVertical[(row - mBoxH - 2) * mWidth + col]) / (len - 1);

                    if (row + 1 < mHeight)
                    {
                        dst[(row + 1) * mWidth + col] = (dst[(row - 1) * mWidth + col] * len - srcVertical[(row - mBoxH - 1) * mWidth + col]) / (len - 1);
                    }

                    len--;
                }
            }
        }
    }
}
