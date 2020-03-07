/*
 * This file is part of the rc_genicam_api package.
 *
 * Copyright (c) 2017 Roboception GmbH
 * All rights reserved
 *
 * Author: Heiko Hirschmueller
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "image.h"

#include "exception.h"
#include "pixel_formats.h"

#include <cstring>

namespace rcg
{

Image::Image(const Buffer *buffer, std::uint32_t part)
{
  if (buffer->getImagePresent(part))
  {
    timestamp=buffer->getTimestampNS();

    width=buffer->getWidth(part);
    height=buffer->getHeight(part);
    xoffset=buffer->getXOffset(part);
    yoffset=buffer->getYOffset(part);
    xpadding=buffer->getXPadding(part);
    ypadding=buffer->getYPadding();
    frameid=buffer->getFrameID();
    pixelformat=buffer->getPixelFormat(part);
    bigendian=buffer->isBigEndian();

    const size_t size=buffer->getSize(part);

    pixel.reset(new uint8_t [size]);

    memcpy(pixel.get(), reinterpret_cast<uint8_t *>(buffer->getBase(part)), size);
  }
  else
  {
    throw GenTLException("Image::Image(): Now image available.");
  }
}

namespace
{

/**
  Clamp the given value to the range of 0 to 255 and cast to byte.
*/

inline unsigned char clamp8(int v)
{
  const int v2=v<0 ? 0:v;
  return static_cast<unsigned char>(v2>255 ? 255:v2);
}

}

void convYCbCr411toRGB(uint8_t rgb[3], const uint8_t *row, int i)
{
  const uint32_t j=static_cast<uint32_t>((i>>2)*6);
  const uint32_t js=static_cast<uint32_t>(i&0x3);

  int Y=row[j+js];
  if (js > 1)
  {
    Y=row[j+js+1];
  }

  const int Cb=static_cast<int>(row[j+2])-128;
  const int Cr=static_cast<int>(row[j+5])-128;

  const int rc=(90*Cr+32)>>6;
  const int gc=(-22*Cb-46*Cr+32)>>6;
  const int bc=(113*Cb+32)>>6;

  rgb[0]=clamp8(Y+rc);
  rgb[1]=clamp8(Y+gc);
  rgb[2]=clamp8(Y+bc);
}

void convYCbCr411toQuadRGB(uint8_t rgb[12], const uint8_t *row, int i)
{
  i=(i>>2)*6;

  const int Y[4]={row[i], row[i+1], row[i+3], row[i+4]};
  const int Cb=static_cast<int>(row[i+2])-128;
  const int Cr=static_cast<int>(row[i+5])-128;

  const int rc=(90*Cr+32)>>6;
  const int gc=(-22*Cb-46*Cr+32)>>6;
  const int bc=(113*Cb+32)>>6;

  for (int j=0; j<4; j++)
  {
    *rgb++=clamp8(Y[j]+rc);
    *rgb++=clamp8(Y[j]+gc);
    *rgb++=clamp8(Y[j]+bc);
  }
}

void getColor(uint8_t rgb[3], const std::shared_ptr<const rcg::Image> &img,
              uint32_t ds, uint32_t i, uint32_t k)
{
  i*=ds;
  k*=ds;

  if (img->getPixelFormat() == Mono8) // convert from monochrome
  {
    size_t lstep=img->getWidth()+img->getXPadding();
    const uint8_t *p=img->getPixels()+k*lstep+i;

    uint32_t g=0, n=0;

    for (uint32_t kk=0; kk<ds; kk++)
    {
      for (uint32_t ii=0; ii<ds; ii++)
      {
        g+=p[ii];
        n++;
      }

      p+=lstep;
    }

    rgb[2]=rgb[1]=rgb[0]=static_cast<uint8_t>(g/n);
  }
  else if (img->getPixelFormat() == YCbCr411_8) // convert from YUV
  {
    size_t lstep=(img->getWidth()>>2)*6+img->getXPadding();
    const uint8_t *p=img->getPixels()+k*lstep;

    uint32_t r=0;
    uint32_t g=0;
    uint32_t b=0;
    uint32_t n=0;

    for (uint32_t kk=0; kk<ds; kk++)
    {
      for (uint32_t ii=0; ii<ds; ii++)
      {
        uint8_t v[3];
        rcg::convYCbCr411toRGB(v, p, static_cast<int>(i+ii));

        r+=v[0];
        g+=v[1];
        b+=v[2];
        n++;
      }

      p+=lstep;
    }

    rgb[0]=static_cast<uint8_t>(r/n);
    rgb[1]=static_cast<uint8_t>(g/n);
    rgb[2]=static_cast<uint8_t>(b/n);
  }
}

}
