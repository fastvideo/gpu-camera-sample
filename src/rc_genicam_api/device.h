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

#ifndef RC_GENICAM_API_DEVICE
#define RC_GENICAM_API_DEVICE

#include "interface.h"

#include <mutex>

namespace rcg
{

class Stream;

/**
  The device class encapsulates a Genicam device.

  NOTE: A GenTLException is thrown in case of a severe error.
*/

class Device : public std::enable_shared_from_this<Device>
{
  public:

    enum ACCESS {READONLY, CONTROL, EXCLUSIVE};

    /**
      Constructs a device class. Devices must only be created by the
      interface class.
    */

    Device(const std::shared_ptr<Interface> &parent,
           const std::shared_ptr<const GenTLWrapper> &gentl, const char *id);
    ~Device();

    /**
      Returns the pointer to the parent interface object.

      @return Pointer to parent interface object.
    */

    std::shared_ptr<Interface> getParent() const;

    /**
      Get the internal ID of this device.

      @return ID.
    */

    const std::string &getID() const;

    /**
      Opens the device for working with it. The interface may be opened
      multiple times. However, for each open(), the close() method must be
      called as well.
    */

    void open(ACCESS access);

    /**
      Closes the device. Each call of open() must be followed by a call to
      close() at some point in time.
    */

    void close();

    /**
      Returns the currently available streams of this device.

      NOTE: The device must have been opened with the open() call before
      calling this method.

      @return List of streams.
    */

    std::vector<std::shared_ptr<Stream> > getStreams();

    /**
      Returns the vendor of the device.

      NOTE: This method only returns a non empty string if the parent interface
      has NOT been closed after Interface::getDevice() or Interface::getDevices()
      call, or if this device itself has been opened with the open() call.

      @return Vendor.
    */

    std::string getVendor();

    /**
      Returns the model of the device.

      NOTE: This method only returns a non empty string if the parent interface
      has NOT been closed after Interface::getDevice() or Interface::getDevices()
      call, or if this device itself has been opened with the open() call.

      @return Model.
    */

    std::string getModel();

    /**
      Returns the transport layer type of the device.

      NOTE: This method only returns a non empty string if the parent interface
      has NOT been closed after Interface::getDevice() or Interface::getDevices()
      call, or if this device itself has been opened with the open() call.

      @return Transport layer type.
    */

    std::string getTLType();

    /**
      Returns the display name of the device.

      NOTE: This method only returns a non empty string if the parent interface
      has NOT been closed after Interface::getDevice() or Interface::getDevices()
      call, or if this device itself has been opened with the open() call.

      @return Display name.
    */

    std::string getDisplayName();

    /**
      Returns the access status of the device.

      NOTE: This method only returns a non empty string if the parent interface
      has NOT been closed after Interface::getDevice() or Interface::getDevices()
      call, or if this device itself has been opened with the open() call.

      @return Access status.
    */

    std::string getAccessStatus();

    /**
      Returns the user defined name of the device.

      NOTE: This method only returns a non empty string if the parent interface
      has NOT been closed after Interface::getDevice() or Interface::getDevices()
      call, or if this device itself has been opened with the open() call.

      @return User defined name.
    */

    std::string getUserDefinedName();

    /**
      Returns the serial number of the device.

      NOTE: This method only returns a non empty string if the parent interface
      has NOT been closed after Interface::getDevice() or Interface::getDevices()
      call, or if this device itself has been opened with the open() call.

      @return Serial number.
    */

    std::string getSerialNumber();

    /**
      Returns the version of the device.

      NOTE: This method only returns a non empty string if the parent interface
      has NOT been closed after Interface::getDevice() or Interface::getDevices()
      call, or if this device itself has been opened with the open() call.

      @return Version.
    */

    std::string getVersion();

    /**
      Returns the timestamp frequency of the device.

      NOTE: This method only returns a non empty string if the parent interface
      has NOT been closed after Interface::getDevice() or Interface::getDevices()
      call, or if this device itself has been opened with the open() call.

      @return Tick-frequency of the time stamp clock.
    */

    uint64_t getTimestampFrequency();

    /**
      Returns the node map of this object.

      NOTE: open() must be called before calling this method. The returned
      pointer remains valid until close() of this object is called.

      @return Node map of this object.
    */

    std::shared_ptr<GenApi::CNodeMapRef> getNodeMap();

    /**
      Returns the node map of the remote device.

      NOTE: open() must be called before calling this method. The returned
      pointer remains valid until close() of this object is called.

      @param xml Path and name for storing the received XML file or 0 if xml
                 file should not be stored.
      @return    Node map of this object.
    */

    std::shared_ptr<GenApi::CNodeMapRef> getRemoteNodeMap(const char *xml=0);

    /**
      Get internal interface handle.

      @return Internal handle.
    */

    void *getHandle() const;

  private:

    Device(class Device &); // forbidden
    Device &operator=(const Device &); // forbidden

    std::shared_ptr<Interface> parent;
    std::shared_ptr<const GenTLWrapper> gentl;
    std::string id;

    std::mutex mtx;

    int n_open;
    void *dev;
    void *rp;

    std::shared_ptr<CPort> cport, rport;
    std::shared_ptr<GenApi::CNodeMapRef> nodemap, rnodemap;

    std::vector<std::weak_ptr<Stream> > slist;
};

/**
  Returns a list of all devices that are available across all transport layers
  and interfaces.

  @return List of available devices.
*/

std::vector<std::shared_ptr<Device> > getDevices();

/**
  Searches across all transport layers and interfaces for a device. This method
  accepts optionally specifying the interface ID as prefix, followed by ':',
  i.e. "[<interfaca_id>[:]]<device_id>". If the interface ID is not given, then
  all interfaces are sought and the first device with the given ID returned.

  @param devid Device ID.
  @return      Device or null pointer.
*/

std::shared_ptr<Device> getDevice(const char *devid);

}

#endif
