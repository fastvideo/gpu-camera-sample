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

#include "stream.h"

#include "gentl_wrapper.h"
#include "exception.h"
#include "cport.h"

#include <iostream>
#include <algorithm>

#ifdef _WIN32
#undef min
#undef max
#endif

namespace rcg
{

Stream::Stream(const std::shared_ptr<Device> &_parent,
               const std::shared_ptr<const GenTLWrapper> &_gentl, const char *_id) :
               buffer(_gentl, this)
{
  parent=_parent;
  gentl=_gentl;
  id=_id;

  n_open=0;
  stream=0;
  event=0;
  bn=0;
}

Stream::~Stream()
{
  try
  {
    stopStreaming();

    if (stream != 0)
    {
      gentl->DSClose(stream);
    }
  }
  catch (...) // do not throw exceptions in destructor
  { }
}

std::shared_ptr<Device> Stream::getParent() const
{
  return parent;
}

const std::string &Stream::getID() const
{
  return id;
}

void Stream::open()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);

  if (n_open == 0)
  {
    if (parent->getHandle() != 0)
    {
      if (gentl->DevOpenDataStream(parent->getHandle(), id.c_str(), &stream) !=
          GenTL::GC_ERR_SUCCESS)
      {
        throw GenTLException("Stream::open()", gentl);
      }
    }
    else
    {
      throw GenTLException("Stream::open(): Device must be opened before open before opening a stream");
    }
  }

  n_open++;
}

void Stream::close()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);

  if (n_open > 0)
  {
    n_open--;
  }

  if (n_open == 0)
  {
    stopStreaming();
    gentl->DSClose(stream);
    stream=0;

    nodemap=0;
    cport=0;
  }
}

void Stream::startStreaming(int na)
{
  std::lock_guard<std::recursive_mutex> lock(mtx);

  buffer.setHandle(0);

  if (stream == 0)
  {
    throw GenTLException("Stream::startStreaming(): Stream is not open");
  }

  // stop streaming if it is currently running

  if (bn > 0)
  {
    stopStreaming();
  }

  // lock parameters before streaming starts

  std::shared_ptr<GenApi::CNodeMapRef> nmap=parent->getRemoteNodeMap();
  GenApi::IInteger *p=dynamic_cast<GenApi::IInteger *>(nmap->_GetNode("TLParamsLocked"));

  if (GenApi::IsWritable(p))
  {
    p->SetValue(1);
  }

  // determine maximum buffer size from transport layer or remote device

  size_t size=0;
  if (getDefinesPayloadsize())
  {
    size=getPayloadSize();
  }
  else
  {
    std::shared_ptr<GenApi::CNodeMapRef> nmap=parent->getRemoteNodeMap();
    GenApi::IInteger *p=dynamic_cast<GenApi::IInteger *>(nmap->_GetNode("PayloadSize"));

    if (GenApi::IsReadable(p))
    {
      size=static_cast<size_t>(p->GetValue());
    }
  }

  // announce and queue the minimum number of buffers

  bool err=false;

  bn=std::max(static_cast<size_t>(8), getBufAnnounceMin());
  for (size_t i=0; i<bn; i++)
  {
    GenTL::BUFFER_HANDLE p=0;

    if (gentl->DSAllocAndAnnounceBuffer(stream, size, 0, &p) != GenTL::GC_ERR_SUCCESS)
    {
      err=true;
      break;
    }

    GenTL::GC_ERROR ret = gentl->DSQueueBuffer(stream, p);
    if (!err && ret != GenTL::GC_ERR_SUCCESS)
    {
      err=true;
      break;
    }
  }

  // register event

  if (!err && gentl->GCRegisterEvent(stream, GenTL::EVENT_NEW_BUFFER, &event) !=
      GenTL::GC_ERR_SUCCESS)
  {
    err=true;
  }

  // start streaming

  uint64_t n=GENTL_INFINITE;

  if (na > 0)
  {
    n=static_cast<uint64_t>(na);
  }

  if (!err && gentl->DSStartAcquisition(stream, GenTL::ACQ_START_FLAGS_DEFAULT, n) !=
      GenTL::GC_ERR_SUCCESS)
  {
    gentl->GCUnregisterEvent(stream, GenTL::EVENT_NEW_BUFFER);
    err=true;
  }

  if (!err)
  {
    GenApi::CCommandPtr start=parent->getRemoteNodeMap()->_GetNode("AcquisitionStart");
    start->Execute();
  }

  // revoke buffers in case of an error, before throwing an event

  if (err)
  {
    gentl->DSFlushQueue(stream, GenTL::ACQ_QUEUE_ALL_DISCARD);

    GenTL::BUFFER_HANDLE p=0;
    while (gentl->DSGetBufferID(stream, 0, &p) == GenTL::GC_ERR_SUCCESS)
    {
      gentl->DSRevokeBuffer(stream, p, 0, 0);
    }

    // unlock parameters

    std::shared_ptr<GenApi::CNodeMapRef> nmap=parent->getRemoteNodeMap();
    GenApi::IInteger *pi=dynamic_cast<GenApi::IInteger *>(nmap->_GetNode("TLParamsLocked"));

    if (GenApi::IsWritable(pi))
    {
      pi->SetValue(0);
    }

    throw GenTLException("Stream::startStreaming()", gentl);
  }
}

void Stream::stopStreaming()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);

  if (bn > 0)
  {
    buffer.setHandle(0);

    // do not throw exceptions as this method is also called in destructor

    GenApi::CCommandPtr stop=parent->getRemoteNodeMap()->_GetNode("AcquisitionStop");
    stop->Execute();

    gentl->DSStopAcquisition(stream, GenTL::ACQ_STOP_FLAGS_DEFAULT);
    gentl->GCUnregisterEvent(stream, GenTL::EVENT_NEW_BUFFER);
    gentl->DSFlushQueue(stream, GenTL::ACQ_QUEUE_ALL_DISCARD);

    // free all buffers

    for (size_t i=0; i<bn; i++)
    {
      GenTL::BUFFER_HANDLE p=0;
      if (gentl->DSGetBufferID(stream, 0, &p) == GenTL::GC_ERR_SUCCESS)
      {
        gentl->DSRevokeBuffer(stream, p, 0, 0);
      }
    }

    event=0;
    bn=0;

    // unlock parameters

    std::shared_ptr<GenApi::CNodeMapRef> nmap=parent->getRemoteNodeMap();
    GenApi::IInteger *pi=dynamic_cast<GenApi::IInteger *>(nmap->_GetNode("TLParamsLocked"));

    if (GenApi::IsWritable(pi))
    {
      pi->SetValue(0);
    }
  }
}

const Buffer *Stream::grab(int64_t _timeout)
{
  std::lock_guard<std::recursive_mutex> lock(mtx);

  uint64_t timeout=GENTL_INFINITE;
  if (_timeout >= 0)
  {
    timeout=static_cast<uint64_t>(_timeout);
  }

  // check that streaming had been started

  if (bn == 0 && event == 0)
  {
    throw GenTLException("Streaming::grab(): Streaming not started");
  }

  // enqueue previously delivered buffer if any

  if (buffer.getHandle() != 0)
  {
    if (gentl->DSQueueBuffer(stream, buffer.getHandle()) != GenTL::GC_ERR_SUCCESS)
    {
      buffer.setHandle(0);
      throw GenTLException("Stream::grab()", gentl);
    }

    buffer.setHandle(0);
  }

  // wait for event

  GenTL::EVENT_NEW_BUFFER_DATA data;
  size_t size=sizeof(GenTL::EVENT_NEW_BUFFER_DATA);
  memset(&data, 0, size);

  GenTL::GC_ERROR err=gentl->EventGetData(event, &data, &size, timeout);

  // return 0 in case of abort and timeout and throw exception in case of
  // another error

  if (err == GenTL::GC_ERR_ABORT || err == GenTL::GC_ERR_TIMEOUT)
  {
    return 0;
  }
  else if (err != GenTL::GC_ERR_SUCCESS)
  {
    throw GenTLException("Stream::grab()", gentl);
  }

  // return buffer

  buffer.setHandle(data.BufferHandle);

  return &buffer;
}

namespace
{

template<class T> inline T getStreamValue(const std::shared_ptr<const GenTLWrapper> &gentl,
                                          void *stream, GenTL::STREAM_INFO_CMD cmd)
{
  T ret=0;

  GenTL::INFO_DATATYPE type;
  size_t size=sizeof(T);

  if (stream != 0)
  {
    gentl->DSGetInfo(stream, cmd, &type, &ret, &size);
  }

  return ret;
}

inline bool getStreamBool(const std::shared_ptr<const GenTLWrapper> &gentl,
                          void *stream, GenTL::STREAM_INFO_CMD cmd)
{
  bool8_t ret=0;

  GenTL::INFO_DATATYPE type;
  size_t size=sizeof(ret);

  if (stream != 0)
  {
    gentl->DSGetInfo(stream, cmd, &type, &ret, &size);
  }

  return ret != 0;
}

}

uint64_t Stream::getNumDelivered()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);
  return getStreamValue<uint64_t>(gentl, stream, GenTL::STREAM_INFO_NUM_DELIVERED);
}

uint64_t Stream::getNumUnderrun()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);
  return getStreamValue<uint64_t>(gentl, stream, GenTL::STREAM_INFO_NUM_UNDERRUN);
}

size_t Stream::getNumAnnounced()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);
  return getStreamValue<size_t>(gentl, stream, GenTL::STREAM_INFO_NUM_ANNOUNCED);
}

size_t Stream::getNumQueued()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);
  return getStreamValue<size_t>(gentl, stream, GenTL::STREAM_INFO_NUM_QUEUED);
}

size_t Stream::getNumAwaitDelivery()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);
  return getStreamValue<size_t>(gentl, stream, GenTL::STREAM_INFO_NUM_AWAIT_DELIVERY);
}

uint64_t Stream::getNumStarted()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);
  return getStreamValue<uint64_t>(gentl, stream, GenTL::STREAM_INFO_NUM_STARTED);
}

size_t Stream::getPayloadSize()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);
  return getStreamValue<size_t>(gentl, stream, GenTL::STREAM_INFO_PAYLOAD_SIZE);
}

bool Stream::getIsGrabbing()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);
  return getStreamBool(gentl, stream, GenTL::STREAM_INFO_IS_GRABBING);
}

bool Stream::getDefinesPayloadsize()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);
  return getStreamBool(gentl, stream, GenTL::STREAM_INFO_DEFINES_PAYLOADSIZE);
}

std::string Stream::getTLType()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);
  std::string ret;

  GenTL::INFO_DATATYPE type;
  char tmp[1024]="";
  size_t size=sizeof(tmp);

  if (stream != 0)
  {
    if (gentl->DSGetInfo(stream, GenTL::STREAM_INFO_TLTYPE, &type, &ret, &size) ==
        GenTL::GC_ERR_SUCCESS)
    {
      if (type == GenTL::INFO_DATATYPE_STRING)
      {
        ret=tmp;
      }
    }
  }

  return ret;
}

size_t Stream::getNumChunksMax()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);
  return getStreamValue<size_t>(gentl, stream, GenTL::STREAM_INFO_NUM_CHUNKS_MAX);
}

size_t Stream::getBufAnnounceMin()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);
  return getStreamValue<size_t>(gentl, stream, GenTL::STREAM_INFO_BUF_ANNOUNCE_MIN);
}

size_t Stream::getBufAlignment()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);
  return getStreamValue<size_t>(gentl, stream, GenTL::STREAM_INFO_BUF_ALIGNMENT);
}

std::shared_ptr<GenApi::CNodeMapRef> Stream::getNodeMap()
{
  std::lock_guard<std::recursive_mutex> lock(mtx);
  if (stream != 0 && !nodemap)
  {
    cport=std::shared_ptr<CPort>(new CPort(gentl, &stream));
    nodemap=allocNodeMap(gentl, stream, cport.get());
  }

  return nodemap;
}

void *Stream::getHandle() const
{
  return stream;
}

}
