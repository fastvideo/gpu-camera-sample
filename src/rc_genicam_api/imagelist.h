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

#ifndef RC_GENICAM_API_IMAGELIST
#define RC_GENICAM_API_IMAGELIST

#include "image.h"

#include <memory>
#include <vector>

namespace rcg
{

/**
  An object of this class manages a limited number of images. It is intended as
  a helper class for time synchronization of different images that can be
  associated by timestamp.
*/

class ImageList
{
  public:

    /**
      Create an image list.

      @param maxsize Maximum number of elements that the list can hold. The
                     default is 25, which is 1s at 25 Hz.
    */

    ImageList(size_t maxsize=25);

    /**
      Adds the given image to the internal list. If the maximum number of
      elements is exceeded, then the oldes image will be dropped.

      @param image Image to be added.
    */

    void add(const std::shared_ptr<const Image> &image);

    /**
      Creates an image from the given buffer and adds it to the internal list.
      If the maximum number of elements is exceeded, then the oldes image will
      be dropped.

      @param buffer Buffer from which an image will be created.
      @param part   Part number from which the image should be created.
    */

    void add(const Buffer *buffer, uint32_t part);

    /**
      Removes all images that have a timestamp that is older or equal than the
      given timestamp.

      @param timestamp Timestamp.
    */

    void removeOld(uint64_t timestamp);

    /**
      Get oldest timestamp of the list.

      @return Oldest timestamp available or 0 if list is empty.
    */

    uint64_t getOldestTime() const;

    /**
      Returns the image that has the given timestamp. If the image cannot be
      found, then a nullptr is returned.

      @param timestamp Timestamp.
      @return Pointer to image or 0.
    */

    std::shared_ptr<const Image> find(uint64_t timestamp) const;

    /**
      Returns the oldest image that has a timestamp within the tolerance of the
      given timestamp. If the tolerance is <= 0, then the behaviour is the same
      as for find(timestamp). If the image cannot be found, then a nullptr is
      returned.

      @param timestamp Timestamp.
      @param tolerance Maximum tolarance added or subtracted to the timestamp.
      @return Pointer to image or 0.
    */

    std::shared_ptr<const Image> find(uint64_t timestamp,
                                      uint64_t tolerance) const;

  private:

    size_t maxsize;
    std::vector<std::shared_ptr<const Image> > list;
};

}

#endif
