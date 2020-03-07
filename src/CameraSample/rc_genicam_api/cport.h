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

#ifndef RC_GENICAM_API_CPORT
#define RC_GENICAM_API_CPORT

#include "gentl_wrapper.h"

#include <GenApi/GenApi.h>

namespace rcg
{

/**
  This is the port definition that connects GenAPI to GenTL. It is implemented
  such that it works with a pointer to a handle. The methods do nothing if the
  handle is 0.
*/

class CPort : public GenApi::IPort
{
  public:

    CPort(std::shared_ptr<const GenTLWrapper> gentl, void **port);

    void Read(void *buffer, int64_t addr, int64_t length);
    void Write(const void *buffer, int64_t addr, int64_t length);
    GenApi::EAccessMode GetAccessMode() const;

  private:

    std::shared_ptr<const GenTLWrapper> gentl;
    void **port;
};

/**
  Convenience function that returns a GenICam node map from the given port.

  @param gentl Pointer to GenTL Wrapper.
  @param port  Pointer to module or remote port.
  @param cport Pointer to CPort Wrapper.
  @param xml   Path and name for storing the received XML file or 0 if xml file
               should not be stored.
  @return      Allocated node map object or 0 if it cannot be allocated. The
               pointer must be freed by the calling function with delete.
*/

std::shared_ptr<GenApi::CNodeMapRef> allocNodeMap(std::shared_ptr<const GenTLWrapper> gentl,
                                                  void *port, CPort *cport, const char *xml=0);

}

#endif