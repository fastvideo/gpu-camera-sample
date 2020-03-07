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

#ifndef RC_GENICAM_API_STREAM
#define RC_GENICAM_API_STREAM

#include "device.h"
#include "buffer.h"

#include <mutex>

namespace rcg
{

class Buffer;

/**
  The stream class encapsulates a Genicam stream.

  NOTE: A GenTLException is thrown in case of a severe error.
*/

class Stream : public std::enable_shared_from_this<Stream>
{
  public:

    /**
      Constructs a stream class. Streams must only be created by the
      device class.
    */

    Stream(const std::shared_ptr<Device> &parent,
           const std::shared_ptr<const GenTLWrapper> &gentl, const char *id);
    ~Stream();

    /**
      Returns the pointer to the parent device object.

      @return Pointer to parent device object.
    */

    std::shared_ptr<Device> getParent() const;

    /**
      Get the internal ID of this stream.

      @return ID.
    */

    const std::string &getID() const;

    /**
      Opens the stream for working with it. The stream may be opened
      multiple times. However, for each open(), the close() method must be
      called as well.
    */

    void open();

    /**
      Closes the stream. Each call of open() must be followed by a call to
      close() at some point in time.
    */

    void close();

    /**
      Allocates buffers and registers internal events if necessary and starts
      streaming.

      @param na Number of buffers to acquire. Set <= 0 for infinity.
    */

    void startStreaming(int na=-1);

    /**
      Stops streaming.
    */

    void stopStreaming();

    /**
      Wait for the next image or data and return it in a buffer object. The
      buffer is valid until the next call to grab.

      @param timeout Timeout in ms. A value < 0 sets waiting time to infinite.
      @return        Pointer to received buffer or 0 in case of an error or
                     interrupt.
    */

    const Buffer *grab(int64_t timeout=-1);

    /**
      Returns some information about the stream.

      NOTE: open() must have called before calling this method.

      @return Number of delivered buffers since last acquisition start.
    */

    uint64_t getNumDelivered();

    /**
      Returns some information about the stream.

      NOTE: open() must have called before calling this method.

      @return Number of lost buffers due to queue underrun.
    */

    uint64_t getNumUnderrun();

    /**
      Returns some information about the stream.

      NOTE: open() must have called before calling this method.

      @return Number of announced buffers.
    */

    size_t getNumAnnounced();

    /**
      Returns some information about the stream.

      NOTE: open() must have called before calling this method.

      @return Number of buffers in the input pool.
    */

    size_t getNumQueued();

    /**
      Returns some information about the stream.

      NOTE: open() must have called before calling this method.

      @return Number of buffers in the output queue.
    */

    size_t getNumAwaitDelivery();

    /**
      Returns some information about the stream.

      NOTE: open() must have called before calling this method.

      @return Number of buffers started in the acquisition engine.
    */

    uint64_t getNumStarted();

    /**
      Returns some information about the stream.

      NOTE: open() must have called before calling this method.

      @return Size of the expected data in bytes.
    */

    size_t getPayloadSize();

    /**
      Returns some information about the stream.

      NOTE: open() must have called before calling this method.

      @return Flag indicating whether the acquisition engine is started or not.
    */

    bool getIsGrabbing();

    /**
      Returns some information about the stream.

      NOTE: open() must have called before calling this method.

      @return Flag that indicated that this data stream defines a payload size
              independent from the remote device.
    */

    bool getDefinesPayloadsize();

    /**
      Returns some information about the stream.

      NOTE: open() must have called before calling this method.

      @return Transport layer technology that is supported.
    */

    std::string getTLType();

    /**
      Returns some information about the stream.

      NOTE: open() must have called before calling this method.

      @return Max number of chunks in a buffer, if known.
    */

    size_t getNumChunksMax();

    /**
      Returns some information about the stream.

      NOTE: open() must have called before calling this method.

      @return Min number of buffers to announce before acq can start, if known.
    */

    size_t getBufAnnounceMin();

    /**
      Returns some information about the stream.

      NOTE: open() must have called before calling this method.

      @return Buffer alignment in bytes.
    */

    size_t getBufAlignment();

    /**
      Returns the node map of this object.

      NOTE: open() must be called before calling this method. The returned
      pointer remains valid until close() of this object is called.

      @return Node map of this object.
    */

    std::shared_ptr<GenApi::CNodeMapRef> getNodeMap();

    /**
      Get internal stream handle.

      @return Internal handle.
    */

    void *getHandle() const;

  private:

    Stream(class Stream &); // forbidden
    Stream &operator=(const Stream &); // forbidden

    Buffer buffer;

    std::shared_ptr<Device> parent;
    std::shared_ptr<const GenTLWrapper> gentl;
    std::string id;

    std::recursive_mutex mtx;

    int n_open;
    void *stream;
    void *event;
    size_t bn;

    std::shared_ptr<CPort> cport;
    std::shared_ptr<GenApi::CNodeMapRef> nodemap;
};

}

#endif
