/*
 * This file is part of the rc_genicam_api package.
 *
 * Copyright (c) 2019 Roboception GmbH
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

#include "pointcloud.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>

#ifdef _WIN32
#undef min
#undef max
#endif

namespace rcg
{

namespace
{

/*
  Get the i-th 16 bit value.

  @param p         Pointer to first byte of array of 16 bit values.
  @param bigendian True if values are given as big endian. Otherwise, litte
                   endian is assumed.
  @param i         Index of 16 bit inside the given array.
*/

inline uint16_t getUint16(const uint8_t *p, bool bigendian, size_t i)
{
  uint16_t ret;

  if (bigendian)
  {
    size_t j=i<<1;
    ret=static_cast<uint16_t>(((p[j]<<8)|p[j+1]));
  }
  else
  {
    size_t j=i<<1;
    ret=static_cast<uint16_t>(((p[j+1]<<8)|p[j]));
  }

  return ret;
}

}

void storePointCloud(std::string name, double f, double t, double scale,
                     std::shared_ptr<const Image> left,
                     std::shared_ptr<const Image> disp,
                     std::shared_ptr<const Image> conf,
                     std::shared_ptr<const Image> error)
{
  // get size and scale factor between left image and disparity image

  size_t width=disp->getWidth();
  size_t height=disp->getHeight();
  bool bigendian=disp->isBigEndian();
  size_t ds=(left->getWidth()+disp->getWidth()-1)/disp->getWidth();

  // convert focal length factor into focal length in (disparity) pixels

  f*=width;

  // get pointer to disparity data and size of row in bytes

  const uint8_t *dps=disp->getPixels();
  size_t dstep=disp->getWidth()*sizeof(uint16_t)+disp->getXPadding();

  // count number of valid disparities and store vertice index in a temporary
  // index image

  size_t vi=0;
  const uint32_t vinvalid=0xffffffff;
  std::vector<uint32_t> vindex(width*height);

  uint32_t n=0;
  for (size_t k=0; k<height; k++)
  {
    int j=0;
    for (size_t i=0; i<width; i++)
    {
      vindex[vi]=vinvalid;
      if ((dps[j]|dps[j+1]) != 0) vindex[vi]=n++;

      j+=2;
      vi++;
    }

    dps+=dstep;
  }

  dps=disp->getPixels();

  // count number of triangles

  const uint16_t vstep=static_cast<uint16_t>(std::ceil(2/scale));

  int tn=0;
  for (size_t k=1; k<height; k++)
  {
    for (size_t i=1; i<width; i++)
    {
      uint16_t v[4];
      v[0]=getUint16(dps, bigendian, i-1);
      v[1]=getUint16(dps, bigendian, i);
      v[2]=getUint16(dps+dstep, bigendian, i-1);
      v[3]=getUint16(dps+dstep, bigendian, i);

      uint16_t vmin=65535;
      uint16_t vmax=0;
      int valid=0;

      for (int jj=0; jj<4; jj++)
      {
        if (v[jj])
        {
          vmin=std::min(vmin, v[jj]);
          vmax=std::max(vmax, v[jj]);
          valid++;
        }
      }

      if (valid >= 3 && vmax-vmin <= vstep)
      {
        tn+=valid-2;
      }
    }

    dps+=dstep;
  }

  dps=disp->getPixels();

  // get pointer to optional confidence and error data and size of row in bytes

  const uint8_t *cps=0, *eps=0;
  size_t cstep=0, estep=0;

  if (conf)
  {
    cps=conf->getPixels();
    cstep=conf->getWidth()*sizeof(uint8_t)+conf->getXPadding();
  }

  if (error)
  {
    eps=error->getPixels();
    estep=error->getWidth()*sizeof(uint8_t)+error->getXPadding();
  }

  // open output file and write ASCII PLY header

  if (name.size() == 0)
  {
    std::ostringstream os;
    double timestamp=left->getTimestampNS()/1000000000.0;
    os << "rc_visard_" << std::setprecision(16) << timestamp << ".ply";
    name=os.str();
  }

  std::ofstream out(name);

  out << "ply" << std::endl;
  out << "format ascii 1.0" << std::endl;
  out << "comment Created with gc_pointcloud from Roboception GmbH" << std::endl;
  out << "comment Camera [1 0 0; 0 1 0; 0 0 1] [0 0 0]" << std::endl;
  out << "element vertex " << n << std::endl;
  out << "property float32 x" << std::endl;
  out << "property float32 y" << std::endl;
  out << "property float32 z" << std::endl;
  out << "property float32 scan_size" << std::endl; // i.e. size of 3D point

  if (cps != 0)
  {
    out << "property float32 scan_conf" << std::endl; // optional confidence
  }

  if (eps != 0)
  {
    out << "property float32 scan_error" << std::endl; // optional error in 3D along line of sight
  }

  out << "property uint8 diffuse_red" << std::endl;
  out << "property uint8 diffuse_green" << std::endl;
  out << "property uint8 diffuse_blue" << std::endl;
  out << "element face " << tn << std::endl;
  out << "property list uint8 uint32 vertex_indices" << std::endl;
  out << "end_header" << std::endl;

  // create colored point cloud

  for (size_t k=0; k<height; k++)
  {
    for (size_t i=0; i<width; i++)
    {
      // convert disparity from fixed comma 16 bit integer into float value

      double d=scale*getUint16(dps, bigendian, i);

      // if disparity is valid and color can be obtained

      if (d)
      {
        // reconstruct 3D point from disparity value

        double x=(i+0.5-0.5*width)*t/d;
        double y=(k+0.5-0.5*height)*t/d;
        double z=f*t/d;

        // compute size of reconstructed point

        double x2=(i-0.5*width)*t/d;
        double size=2*1.4*std::abs(x2-x);

        // get corresponding color value

        uint8_t rgb[3];
        getColor(rgb, left, static_cast<uint32_t>(ds), static_cast<uint32_t>(i),
                 static_cast<uint32_t>(k));

        // store colored point, optionally with confidence and error

        out << x << " " << y << " " << z << " " << size << " ";

        if (cps != 0)
        {
          out << cps[i]/255.0 << " ";
        }

        if (eps != 0)
        {
          out << eps[i]*scale*f*t/(d*d) << " ";
        }

        out << static_cast<int>(rgb[0]) << " ";
        out << static_cast<int>(rgb[1]) << " ";
        out << static_cast<int>(rgb[2]) << std::endl;
      }
    }

    dps+=dstep;
    cps+=cstep;
    eps+=estep;
  }

  dps=disp->getPixels();

  // create triangles

  uint32_t *ips=vindex.data();
  for (size_t k=1; k<height; k++)
  {
    for (size_t i=1; i<width; i++)
    {
      uint16_t v[4];
      v[0]=getUint16(dps, bigendian, i-1);
      v[1]=getUint16(dps, bigendian, i);
      v[2]=getUint16(dps+dstep, bigendian, i-1);
      v[3]=getUint16(dps+dstep, bigendian, i);

      uint16_t vmin=65535;
      uint16_t vmax=0;
      int valid=0;

      for (int jj=0; jj<4; jj++)
      {
        if (v[jj])
        {
          vmin=std::min(vmin, v[jj]);
          vmax=std::max(vmax, v[jj]);
          valid++;
        }
      }

      if (valid >= 3 && vmax-vmin <= vstep)
      {
        int j=0;
        uint32_t fc[4];

        if (ips[i-1] != vinvalid)
        {
          fc[j++]=ips[i-1];
        }

        if (ips[width+i-1] != vinvalid)
        {
          fc[j++]=ips[width+i-1];
        }

        if (ips[width+i] != vinvalid)
        {
          fc[j++]=ips[width+i];
        }

        if (ips[i] != vinvalid)
        {
          fc[j++]=ips[i];
        }

        out << "3 " << fc[0] << ' ' << fc[1] << ' ' << fc[2] << std::endl;

        if (j == 4)
        {
          out << "3 " << fc[2] << ' ' << fc[3] << ' ' << fc[0] << std::endl;
        }
      }
    }

    ips+=width;
    dps+=dstep;
  }

  out.close();
}

}
