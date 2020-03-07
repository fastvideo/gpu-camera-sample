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

#ifndef RC_GENICAM_API_BUFFER
#define RC_GENICAM_API_BUFFER

#include <memory>
#include <string>

namespace rcg
{

class Stream;
class GenTLWrapper;

/**
  Payload types as taken from GenTL 1.5 definition.
  See Buffer::getPayloadType().
*/

enum PAYLOADTYPE_INFO_IDS
{
  PAYLOAD_TYPE_UNKNOWN         =  0,   /* GenTL v1.2 */
  PAYLOAD_TYPE_IMAGE           =  1,   /* GenTL v1.2 */
  PAYLOAD_TYPE_RAW_DATA        =  2,   /* GenTL v1.2 */
  PAYLOAD_TYPE_FILE            =  3,   /* GenTL v1.2 */
  PAYLOAD_TYPE_CHUNK_DATA      =  4,   /* GenTL v1.2, Deprecated in GenTL 1.5*/
  PAYLOAD_TYPE_JPEG            =  5,   /* GenTL v1.4 */
  PAYLOAD_TYPE_JPEG2000        =  6,   /* GenTL v1.4 */
  PAYLOAD_TYPE_H264            =  7,   /* GenTL v1.4 */
  PAYLOAD_TYPE_CHUNK_ONLY      =  8,   /* GenTL v1.4 */
  PAYLOAD_TYPE_DEVICE_SPECIFIC =  9,   /* GenTL v1.4 */
  PAYLOAD_TYPE_MULTI_PART      =  10,  /* GenTL v1.5 */

  PAYLOAD_TYPE_CUSTOM_ID       = 1000  /* Starting value for GenTL Producer custom IDs. */
};

/**
  Pixelformat namespace IDs as taken from GenTL 1.5 definition.
  See Buffer::getPixelFormatNamespace().
*/

enum PIXELFORMAT_NAMESPACE_IDS
{
  PIXELFORMAT_NAMESPACE_UNKNOWN      = 0,   /* GenTL v1.2 */
  PIXELFORMAT_NAMESPACE_GEV          = 1,   /* GenTL v1.2 */
  PIXELFORMAT_NAMESPACE_IIDC         = 2,   /* GenTL v1.2 */
  PIXELFORMAT_NAMESPACE_PFNC_16BIT   = 3,   /* GenTL v1.4 */
  PIXELFORMAT_NAMESPACE_PFNC_32BIT   = 4,   /* GenTL v1.4 */

  PIXELFORMAT_NAMESPACE_CUSTOM_ID    = 1000 /* Starting value for GenTL Producer custom IDs. */
};

/**
  Enumeration describing which data type is present in given buffer part as
  taken from GenTL 1.5 definition. See Buffer::getPartDataType().
*/

enum PARTDATATYPE_IDS
{
  PART_DATATYPE_UNKNOWN              =  0,   /* Unknown data type */
  PART_DATATYPE_2D_IMAGE             =  1,   /* Color or monochrome 2D image. */
  PART_DATATYPE_2D_PLANE_BIPLANAR    =  2,   /* Single color plane of a planar 2D image consisting of 2 planes. */
  PART_DATATYPE_2D_PLANE_TRIPLANAR   =  3,   /* Single color plane of a planar 2D image consisting of 3 planes. */
  PART_DATATYPE_2D_PLANE_QUADPLANAR  =  4,   /* Single color plane of a planar 2D image consisting of 4 planes. */
  PART_DATATYPE_3D_IMAGE             =  5,   /* 3D image (pixel coordinates). */
  PART_DATATYPE_3D_PLANE_BIPLANAR    =  6,   /* Single plane of a planar 3D image consisting of 2 planes. */
  PART_DATATYPE_3D_PLANE_TRIPLANAR   =  7,   /* Single plane of a planar 3D image consisting of 3 planes. */
  PART_DATATYPE_3D_PLANE_QUADPLANAR  =  8,   /* Single plane of a planar 3D image consisting of 4 planes. */
  PART_DATATYPE_CONFIDENCE_MAP       =  9,   /* Confidence of the individual pixel values. */
  PART_DATATYPE_CHUNKDATA            = 10,   /* Chunk data type */

  PART_DATATYPE_CUSTOM_ID            = 1000  /* Starting value for GenTL Producer custom IDs. */
};

/**
  The buffer class encapsulates a Genicam buffer that is provided by a stream.
  A multi-part buffer with one image can be treated like a "normal" buffer.

  NOTE: A GenTLException is thrown in case of a severe error.
*/

class Buffer
{
  public:

    /**
      Constructs a buffer class as wrapper around a buffer handle. Buffers must
      only be constructed by the stream class.
    */

    Buffer(const std::shared_ptr<const GenTLWrapper> &gentl, Stream *parent);

    /**
      Set the buffer handle that this object should manage. The handle is used
      until a new handle is set.

      @param handle Buffer handle that replaces a possibly existing handle.
    */

    void setHandle(void *handle);

    /**
      Returns the number of parts, excluding chunk data. This is 1 if the
      buffer is not multipart and the buffer is not chunk only.

      @return Number of parts.
    */

    std::uint32_t getNumberOfParts() const;

    /**
      Returns the global base address of the buffer memory.

      @return Global base address of the buffer memory.
    */

    void *getGlobalBase() const;

    /**
      Returns the global size of the buffer.

      @return Global size of the buffer in bytes.
    */

    size_t getGlobalSize() const;

    /**
      Returns the base address of the specified part of the multi-part buffer.

      @return Base address of the buffer memory part.
    */

    void *getBase(std::uint32_t part) const;

    /**
      Returns the size of the specified part of the mult-part buffer.

      @return Size of the buffer part in bytes.
    */

    size_t getSize(std::uint32_t part) const;

    /**
      Returns the private data pointer of the GenTL Consumer.

      @return Private data pointer of the GenTL Consumer.
    */

    void *getUserPtr() const;

    /**
      Returns the timestamp of the buffer.

      @return Timestamp the buffer was acquired.
    */

    uint64_t getTimestamp() const;

    /**
      Returns if the buffer contains new data.

      @return Flag to indicate that the buffer contains new data since the last
              call.
    */

    bool getNewData() const;

    /**
      Signals if the buffer is associated to the input or output queue.

      @return Flag to indicate if the buffer is in the input pool or output
              queue.
    */

    bool getIsQueued() const;

    /**
      Signals if the buffer is currently being filled with data.

      @return Flag to indicate that the buffer is currently being filled with
              data.
    */

    bool getIsAcquiring() const;

    /**
      Signals if the buffer is incomplete due to an error.

      @return Flag to indicate that a buffer was filled but an error occurred
              during that process.
    */

    bool getIsIncomplete() const;

    /**
      Returns the type the used transport layer.

      @return Transport layer type.
    */

    std::string getTLType() const;

    /**
      Returns the number of bytes written into the buffer last time it has been
      filled. This value is reset to 0 when the buffer is placed into the Input
      Buffer Pool.

      @return Fill size of buffer since last call.
    */

    size_t getSizeFilled() const;

    /**
      Returns the data type id of the specified part as defined in
      PARTDATATYPE_IDS. If this buffer is not mult-part, then 0 is returned.

      @return One of PARTDATATYPE_IDS.
    */

    size_t getPartDataType(uint32_t part) const;

    /**
      Returns the width of the image in pixel.

      @param part Part index, which is ignored if the buffer is not multi-part.
      @return     Width of image.
    */

    size_t getWidth(std::uint32_t part) const;

    /**
      Returns the height of the image in pixel.

      @param part Part index, which is ignored if the buffer is not multi-part.
      @return     Height of image.
    */

    size_t getHeight(std::uint32_t part) const;

    /**
      Returns the horizontal offset of the data in the buffer in pixels from
      the image origin to handle areas of interest.

      @param part Part index, which is ignored if the buffer is not multi-part.
      @return     Horizontal offset of image.
    */

    size_t getXOffset(std::uint32_t part) const;

    /**
      Returns the vertical offset of the data in the buffer in lines from
      the image origin to handle areas of interest.

      @param part Part index, which is ignored if the buffer is not multi-part.
      @return     Vertical offset of image.
    */

    size_t getYOffset(std::uint32_t part) const;

    /**
      Returns horizontal padding of the data in the buffer in bytes.

      @param part Part index, which is ignored if the buffer is not multi-part.
      @return     Horizontal padding of image.
    */

    size_t getXPadding(std::uint32_t part) const;

    /**
      Returns vertical padding of the data in the buffer in bytes.

      @return Vertical padding of image.
    */

    size_t getYPadding() const;

    /**
      Returns the sequentially incremented number of the frame. The wrap around
      depends on the used transport layer technology.

      @return Monotonically increasing frame id.
    */

    uint64_t getFrameID() const;

    /**
      Returns if a 2D, 3D or confidence image is present in the specified part.

      @param part Part index, which is ignored if the buffer is not multi-part.
      @return     True if image is present.
    */

    bool getImagePresent(std::uint32_t part) const;

    /**
      Returns the payload type according to PAYLOADTYPE_INFO_IDS.

      @return One of PAYLOADTYPE_INFO_IDS.
    */

    size_t getPayloadType() const;

    /**
      Returns the pixel format of the specified part as defined in the PFNC.
      The pixel formats are defined in pixel_formats.h and PFNC.h if
      getPixelFormatNamespace() returns PIXELFORMAT_NAMESPACE_PFNC_32BIT.

      @param part Part index, which is ignored if the buffer is not multi-part.
      @return Pixel format id.
    */

    uint64_t getPixelFormat(std::uint32_t part) const;

    /**
      Returns the pixel format namespace, which preferably should be
      PIXELFORMAT_NAMESPACE_PFNC_32BIT.

      @param part Part index, which is ignored if the buffer is not multi-part.
      @return One of PIXELFORMAT_NAMESPACE_IDS.
    */

    uint64_t getPixelFormatNamespace(std::uint32_t part) const;

    /**
      Returns the source id of the specified part. Images with the same source
      id are supposed to belong pixelwise together.

      @param part Part index, which is ignored if the buffer is not multi-part.
      @return     Source id of part or 0 if the buffer is not multi-part.
    */

    uint64_t getPartSourceID(std::uint32_t part) const;

    /**
      Returns the number of lines that are delivered in this buffer. For areea
      cameras, this is typically the same as the specified image height. For
      linescan cameras it can be less.

      @param part Part index, which is ignored if the buffer is not multi-part.
      @return     Number of delivered image lines.
    */

    size_t getDeliveredImageHeight(std::uint32_t part) const;

    /**
      Returnes the delivered chung payload size.

      @return Chunk payload size.
    */

    size_t getDeliveredChunkPayloadSize() const;

    /**
      Returns the chunk layout id, which serves as an indicator that the chunk
      layout has changed and the application should parse the chunk layout
      again.

      @return Chunk layout id.
    */

    uint64_t getChunkLayoutID() const;

    /**
      Returns the filename in case the payload contains a file.

      @return File name.
    */

    std::string getFilename() const;

    /**
      Returns if the data is given as big or little endian.

      @return True for big endian and false for little endian.
    */

    bool isBigEndian() const;

    /**
      Returns the size of data intended to the written to the buffer the last
      time it has been filled. If the buffer is incomplete, the number reports
      the full size of the original data including the lost parts. If the
      buffer is complete, the number equals getSizeFilled().

      @return Data size in bytes.
    */

    size_t getDataSize() const;

    /**
      Returns the acquisition timestamp of the data in this buffer in ns.

      @return Timestamp in ns.
    */

    uint64_t getTimestampNS() const;

    /**
      Signals if the memory that was allocated for this buffer is too small.

      @return True if buffer is too small.
    */

    bool getDataLargerThanBuffer() const;

    /**
      Returns if the buffer contains chunk data.

      @return True if the buffer contains chunk data.
    */

    bool getContainsChunkdata() const;

    /**
      Get internal stream handle.

      @return Internal handle.
    */

    void *getHandle() const;

  private:

    Buffer(class Buffer &); // forbidden
    Buffer &operator=(const Buffer &); // forbidden

    Stream *parent;
    std::shared_ptr<const GenTLWrapper> gentl;
    void *buffer;
    bool multipart;
};

bool isHostBigEndian();

}

#endif
