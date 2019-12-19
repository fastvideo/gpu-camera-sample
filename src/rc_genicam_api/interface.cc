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

#include "interface.h"
#include "device.h"

#include "gentl_wrapper.h"
#include "exception.h"
#include "cport.h"

#include <iostream>

namespace rcg
{

Interface::Interface(const std::shared_ptr<System> &_parent,
                     const std::shared_ptr<const GenTLWrapper> &_gentl, const char *_id)
{
  parent=_parent;
  gentl=_gentl;
  id=_id;

  n_open=0;
  ifh=0;
}

Interface::~Interface()
{
  if (n_open > 0)
  {
    gentl->IFClose(ifh);
    parent->close();
  }
}

std::shared_ptr<System> Interface::getParent() const
{
  return parent;
}

const std::string &Interface::getID() const
{
  return id;
}

void Interface::open()
{
  std::lock_guard<std::mutex> lock(mtx);

  if (n_open == 0)
  {
    parent->open();

    // updating before opening is apparently necessary for some TLs

    gentl->TLUpdateInterfaceList(parent->getHandle(), 0, 10);

    if (gentl->TLOpenInterface(parent->getHandle(), id.c_str(), &ifh) != GenTL::GC_ERR_SUCCESS)
    {
      parent->close();
      throw GenTLException("Interface::open()", gentl);
    }
  }

  n_open++;
}

void Interface::close()
{
  std::lock_guard<std::mutex> lock(mtx);

  if (n_open > 0)
  {
    n_open--;
  }

  if (n_open == 0)
  {
    gentl->IFClose(ifh);
    ifh=0;

    nodemap=0;
    cport=0;

    parent->close();
  }
}

namespace
{

int find(const std::vector<std::shared_ptr<Device> > &list, const std::string &id)
{
  for (size_t i=0; i<list.size(); i++)
  {
    if (list[i]->getID() == id)
    {
      return static_cast<int>(i);
    }
  }

  return -1;
}

}

std::vector<std::shared_ptr<Device> > Interface::getDevices()
{
  std::lock_guard<std::mutex> lock(mtx);

  std::vector<std::shared_ptr<Device> > ret;

  if (ifh != 0)
  {
    // get list of previously requested devices that are still in use

    std::vector<std::shared_ptr<Device> > current;

    for (size_t i=0; i<dlist.size(); i++)
    {
      std::shared_ptr<Device> p=dlist[i].lock();
      if (p)
      {
        current.push_back(p);
      }
    }

    // update available interfaces

    if (gentl->IFUpdateDeviceList(ifh, 0, 100) != GenTL::GC_ERR_SUCCESS)
    {
      throw GenTLException("Interface::getDevices()", gentl);
    }

    // create list of interfaces, using either existing interfaces or
    // instantiating new ones

    uint32_t n=0;
    if (gentl->IFGetNumDevices(ifh, &n) != GenTL::GC_ERR_SUCCESS)
    {
      throw GenTLException("Interface::getDevices()", gentl);
    }

    for (uint32_t i=0; i<n; i++)
    {
      char tmp[256]="";
      size_t size=sizeof(tmp);

      if (gentl->IFGetDeviceID(ifh, i, tmp, &size) != GenTL::GC_ERR_SUCCESS)
      {
        throw GenTLException("Interface::getDevices()", gentl);
      }

      int k=find(current, tmp);

      if (k >= 0)
      {
        ret.push_back(current[static_cast<size_t>(k)]);
      }
      else
      {
        ret.push_back(std::shared_ptr<Device>(new Device(shared_from_this(), gentl, tmp)));
      }
    }

    // update internal list of devices for reusage on next call

    dlist.clear();
    for (size_t i=0; i<ret.size(); i++)
    {
      dlist.push_back(ret[i]);
    }
  }

  return ret;
}

std::shared_ptr<Device> Interface::getDevice(const char *devid)
{
  // get list of all devices

  std::vector<std::shared_ptr<Device> > list=getDevices();

  // find requested device by ID or user defined name

  std::shared_ptr<Device> ret;

  for (size_t i=0; i<list.size(); i++)
  {
    std::shared_ptr<Device> p=list[i];

    if (p && (p->getID() == devid || p->getDisplayName() == devid ||
              p->getSerialNumber() == devid))
    {
      if (ret)
      {
        std::cerr << "There is more than one device with ID, serial number or user defined name: "
                  << devid << std::endl;
        ret=0;
        break;
      }

      ret=p;
    }
  }

  return ret;
}

namespace
{

std::string cIFGetInfo(const Interface *obj,
                       const std::shared_ptr<const GenTLWrapper> &gentl,
                       GenTL::INTERFACE_INFO_CMD info)
{
  std::string ret;

  GenTL::INFO_DATATYPE type=GenTL::INFO_DATATYPE_UNKNOWN;
  char tmp[1024]="";
  size_t tmp_size=sizeof(tmp);
  GenTL::GC_ERROR err=GenTL::GC_ERR_ERROR;

  if (obj->getHandle() != 0)
  {
    err=gentl->IFGetInfo(obj->getHandle(), info, &type, tmp, &tmp_size);
  }
  else if (obj->getParent()->getHandle() != 0)
  {
    err=gentl->TLGetInterfaceInfo(obj->getParent()->getHandle(), obj->getID().c_str(), info,
                                  &type, tmp, &tmp_size);
  }

  if (err == GenTL::GC_ERR_SUCCESS && type == GenTL::INFO_DATATYPE_STRING)
  {
    for (size_t i=0; i<tmp_size && tmp[i] != '\0'; i++)
    {
      ret.push_back(tmp[i]);
    }
  }

  return ret;
}

}

std::string Interface::getDisplayName()
{
  std::lock_guard<std::mutex> lock(mtx);
  return cIFGetInfo(this, gentl, GenTL::INTERFACE_INFO_DISPLAYNAME);
}

std::string Interface::getTLType()
{
  std::lock_guard<std::mutex> lock(mtx);
  return cIFGetInfo(this, gentl, GenTL::INTERFACE_INFO_TLTYPE);
}

std::shared_ptr<GenApi::CNodeMapRef> Interface::getNodeMap()
{
  std::lock_guard<std::mutex> lock(mtx);
  if (ifh != 0 && !nodemap)
  {
    cport=std::shared_ptr<CPort>(new CPort(gentl, &ifh));
    nodemap=allocNodeMap(gentl, ifh, cport.get());
  }

  return nodemap;
}

void *Interface::getHandle() const
{
  return ifh;
}

}
