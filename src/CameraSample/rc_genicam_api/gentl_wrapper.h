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

#ifndef RC_GENICAM_API_GENTL_WRAPPER
#define RC_GENICAM_API_GENTL_WRAPPER

#include <GenTL/GenTL.h>

#include <string>
#include <vector>

namespace rcg
{

/**
  The function uses the given list files of paths that is separated by colons
  or semicolons, depending on the used operating system, and returns all files
  with the suffix .cti.

  @param  Paths, which may be 0.
  @return List of files with suffix cti, including path to that file.
*/

std::vector<std::string> getAvailableGenTLs(const char *paths);

/**
  Wrapper for dynamically loaded GenICam transport layers.
*/

class GenTLWrapper
{
  public:

    /*
      Loads a shared lib and tries to resolve all functions that are defined in
      GenTL Version 1.5.

      NOTE: An object of this class should only be instantiated by System for
      internal usage.

      NOTE: An exception is thrown if the lib cannot be loaded or if not all
      symbols can be resolved.

      @param filename Path and name of GenTL shared library.
    */

    GenTLWrapper(const std::string &filename);
    ~GenTLWrapper();

    // C interface functions of GenTL

    GenTL::PGCGetInfo GCGetInfo;
    GenTL::PGCGetLastError GCGetLastError;
    GenTL::PGCInitLib GCInitLib;
    GenTL::PGCCloseLib GCCloseLib;
    GenTL::PGCReadPort GCReadPort;
    GenTL::PGCWritePort GCWritePort;
    GenTL::PGCGetPortURL GCGetPortURL;
    GenTL::PGCGetPortInfo GCGetPortInfo;

    GenTL::PGCRegisterEvent GCRegisterEvent;
    GenTL::PGCUnregisterEvent GCUnregisterEvent;
    GenTL::PEventGetData EventGetData;
    GenTL::PEventGetDataInfo EventGetDataInfo;
    GenTL::PEventGetInfo EventGetInfo;
    GenTL::PEventFlush EventFlush;
    GenTL::PEventKill EventKill;
    GenTL::PTLOpen TLOpen;
    GenTL::PTLClose TLClose;
    GenTL::PTLGetInfo TLGetInfo;
    GenTL::PTLGetNumInterfaces TLGetNumInterfaces;
    GenTL::PTLGetInterfaceID TLGetInterfaceID;
    GenTL::PTLGetInterfaceInfo TLGetInterfaceInfo;
    GenTL::PTLOpenInterface TLOpenInterface;
    GenTL::PTLUpdateInterfaceList TLUpdateInterfaceList;
    GenTL::PIFClose IFClose;
    GenTL::PIFGetInfo IFGetInfo;
    GenTL::PIFGetNumDevices IFGetNumDevices;
    GenTL::PIFGetDeviceID IFGetDeviceID;
    GenTL::PIFUpdateDeviceList IFUpdateDeviceList;
    GenTL::PIFGetDeviceInfo IFGetDeviceInfo;
    GenTL::PIFOpenDevice IFOpenDevice;

    GenTL::PDevGetPort DevGetPort;
    GenTL::PDevGetNumDataStreams DevGetNumDataStreams;
    GenTL::PDevGetDataStreamID DevGetDataStreamID;
    GenTL::PDevOpenDataStream DevOpenDataStream;
    GenTL::PDevGetInfo DevGetInfo;
    GenTL::PDevClose DevClose;

    GenTL::PDSAnnounceBuffer DSAnnounceBuffer;
    GenTL::PDSAllocAndAnnounceBuffer DSAllocAndAnnounceBuffer;
    GenTL::PDSFlushQueue DSFlushQueue;
    GenTL::PDSStartAcquisition DSStartAcquisition;
    GenTL::PDSStopAcquisition DSStopAcquisition;
    GenTL::PDSGetInfo DSGetInfo;
    GenTL::PDSGetBufferID DSGetBufferID;
    GenTL::PDSClose DSClose;
    GenTL::PDSRevokeBuffer DSRevokeBuffer;
    GenTL::PDSQueueBuffer DSQueueBuffer;
    GenTL::PDSGetBufferInfo DSGetBufferInfo;

    // GenTL v1.1

    GenTL::PGCGetNumPortURLs GCGetNumPortURLs;
    GenTL::PGCGetPortURLInfo GCGetPortURLInfo;
    GenTL::PGCReadPortStacked GCReadPortStacked;
    GenTL::PGCWritePortStacked GCWritePortStacked;

    // GenTL v1.3

    GenTL::PDSGetBufferChunkData DSGetBufferChunkData;

    // GenTL v1.4

    GenTL::PIFGetParentTL IFGetParentTL;
    GenTL::PDevGetParentIF DevGetParentIF;
    GenTL::PDSGetParentDev DSGetParentDev;

    // GenTL v1.5

    GenTL::PDSGetNumBufferParts DSGetNumBufferParts;
    GenTL::PDSGetBufferPartInfo DSGetBufferPartInfo;

  private:

    GenTLWrapper(const GenTLWrapper &); // forbidden
    GenTLWrapper &operator=(const GenTLWrapper &); // forbidden

    void *lib;
};

}

#endif
