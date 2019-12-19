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

#ifndef RC_GENICAM_API_POINTCLOUD
#define RC_GENICAM_API_POINTCLOUD

#include "image.h"

#include <string>
#include <memory>

namespace rcg
{

/*
  Computes a point cloud from the given synchronized left and disparity image
  pair and stores it in ply ascii format.

  @param name    Name of output file. If empty, a standard file name with
                 timestamp is used.
  @param f       Focal length factor (to be multiplicated with image width).
  @param t       Baseline in m.
  @param scale   Disparity scale factor.
  @param left    Left camera image. The image must have format Mono8 or
                 YCbCr411_8.
  @param disp    Corresponding disparity image, possibly downscaled by an
                 integer factor. The image must be in format Coord3D_C16.
  @param conf    Optional corresponding confidence image in the same size as
                 disp. The image must be in format Confidence8.
  @param error   Optional corresponding error image in the same size as disp.
                 The image must be in format Error8.
*/

void storePointCloud(std::string name, double f, double t, double scale,
                     std::shared_ptr<const Image> left,
                     std::shared_ptr<const Image> disp,
                     std::shared_ptr<const Image> conf=0,
                     std::shared_ptr<const Image> error=0);

}

#endif
