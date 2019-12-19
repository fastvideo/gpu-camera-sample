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

#ifndef RC_GENICAM_API_INTERFACE
#define RC_GENICAM_API_INTERFACE

#include "system.h"

#include <mutex>

namespace rcg
{

class Device;

/**
  The interface class encapsulates a Genicam interface.

  NOTE: A GenTLException is thrown in case of a severe error.
*/

class Interface : public std::enable_shared_from_this<Interface>
{
  public:

    /**
      Constructs an interface class. Interfaces must only be created by the
      system class.
    */

    Interface(const std::shared_ptr<System> &parent,
              const std::shared_ptr<const GenTLWrapper> &gentl, const char *id);
    ~Interface();

    /**
      Returns the pointer to the parent system object.

      @return Pointer to parent system object.
    */

    std::shared_ptr<System> getParent() const;

    /**
      Get the internal ID of this interface.

      @return ID.
    */

    const std::string &getID() const;

    /**
      Opens the interface for working with it. The interface may be opened
      multiple times. However, for each open(), the close() method must be
      called as well.
    */

    void open();

    /**
      Closes the interface. Each call of open() must be followed by a call to
      close() at some point in time.
    */

    void close();

    /**
      Returns the currently available devices on this interface.

      NOTE: open() must be called before calling this method.

      @return List of devices.
    */

    std::vector<std::shared_ptr<Device> > getDevices();

    /**
      Returns a device with the given device id.

      NOTE: open() must be called before calling this method.

      @return Pointer to device or std::nullptr.
    */

    std::shared_ptr<Device> getDevice(const char *devid);

    /**
      Returns the display name of the interface.

      NOTE: At least the parent object must have been opened before calling
      this method.

      @return Display name.
    */

    std::string getDisplayName();

    /**
      Returns the transport layer type of the interface.

      NOTE: At least the parent object must have been opened before calling
      this method.

      @return Transport layer type.
    */

    std::string getTLType();

    /**
      Returns the node map of this object.

      NOTE: open() must be called before calling this method. The returned
      pointer remains valid until close() of this object is called.

      @return Node map of this object.
    */

    std::shared_ptr<GenApi::CNodeMapRef> getNodeMap();

    /**
      Get internal interace handle.

      @return Internal handle.
    */

    void *getHandle() const;

  private:

    Interface(class Interface &); // forbidden
    Interface &operator=(const Interface &); // forbidden

    std::shared_ptr<System> parent;
    std::shared_ptr<const GenTLWrapper> gentl;
    std::string id;

    std::mutex mtx;

    int n_open;
    void *ifh;

    std::shared_ptr<CPort> cport;
    std::shared_ptr<GenApi::CNodeMapRef> nodemap;

    std::vector<std::weak_ptr<Device> > dlist;
};

}

#endif
