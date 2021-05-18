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

#include "gentl_wrapper.h"

#include <string>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <iostream>

#include <Windows.h>

namespace rcg
{

std::vector<std::string> getAvailableGenTLs(const char *paths)
{
  std::vector<std::string> ret;

  if (paths != 0)
  {
    // split path list into individual paths

    std::stringstream in(paths);
    std::string path;

    while (getline(in, path, ';'))
    {
      if (path.size() > 0)
      {
        if (path.size() > 4 && path.compare(path.size()-4, 4, ".cti") == 0)
        {
          // the given path points to one file ending with .cti

          ret.push_back(path);
        }
        else
        {
          // get all files with suffix .cti

          HANDLE p;
          WIN32_FIND_DATA data;

          std::string dir=path;

          if (dir.size() > 0 && dir[dir.size()-1] != '\\')
          {
            dir+="\\";
          }

          p=FindFirstFileA((dir+"*.cti").c_str(), &data);

          if (p != INVALID_HANDLE_VALUE)
          {
            do
            {
              ret.push_back(dir+data.cFileName);
            }
            while (FindNextFileA(p, &data));

            FindClose(p);
          }
        }
      }
    }
  }

  return ret;
}

namespace
{

inline FARPROC getFunction(HMODULE lib, const char *name)
{
  FARPROC ret=GetProcAddress(lib, name);

  if (ret == 0)
  {
    DWORD err=GetLastError();

    FreeLibrary(lib);

    std::ostringstream out;
    out << "Cannot resolve GenTL symbol. Error code: " << err;

    throw std::invalid_argument(out.str());
  }

  return ret;
}

}

GenTLWrapper::GenTLWrapper(const std::string &filename)
{
  // open library

  HMODULE lp=LoadLibrary(filename.c_str());

  if (lp == 0)
  {
    DWORD err=GetLastError();

    std::ostringstream out;
    out << "Cannot open GenTL library. Error code: " << err;

    throw std::invalid_argument(out.str());
  }

  // resolve function calls that will only be used privately

  *reinterpret_cast<void**>(&GCInitLib)=getFunction(lp, "GCInitLib");
  *reinterpret_cast<void**>(&GCCloseLib)=getFunction(lp, "GCCloseLib");

  // resolve public symbols

  *reinterpret_cast<void**>(&GCGetInfo)=getFunction(lp, "GCGetInfo");
  *reinterpret_cast<void**>(&GCGetLastError)=getFunction(lp, "GCGetLastError");
  *reinterpret_cast<void**>(&GCReadPort)=getFunction(lp, "GCReadPort");
  *reinterpret_cast<void**>(&GCWritePort)=getFunction(lp, "GCWritePort");
  *reinterpret_cast<void**>(&GCGetPortURL)=getFunction(lp, "GCGetPortURL");
  *reinterpret_cast<void**>(&GCGetPortInfo)=getFunction(lp, "GCGetPortInfo");

  *reinterpret_cast<void**>(&GCRegisterEvent)=getFunction(lp, "GCRegisterEvent");
  *reinterpret_cast<void**>(&GCUnregisterEvent)=getFunction(lp, "GCUnregisterEvent");
  *reinterpret_cast<void**>(&EventGetData)=getFunction(lp, "EventGetData");
  *reinterpret_cast<void**>(&EventGetDataInfo)=getFunction(lp, "EventGetDataInfo");
  *reinterpret_cast<void**>(&EventGetInfo)=getFunction(lp, "EventGetInfo");
  *reinterpret_cast<void**>(&EventFlush)=getFunction(lp, "EventFlush");
  *reinterpret_cast<void**>(&EventKill)=getFunction(lp, "EventKill");
  *reinterpret_cast<void**>(&TLOpen)=getFunction(lp, "TLOpen");
  *reinterpret_cast<void**>(&TLClose)=getFunction(lp, "TLClose");
  *reinterpret_cast<void**>(&TLGetInfo)=getFunction(lp, "TLGetInfo");
  *reinterpret_cast<void**>(&TLGetNumInterfaces)=getFunction(lp, "TLGetNumInterfaces");
  *reinterpret_cast<void**>(&TLGetInterfaceID)=getFunction(lp, "TLGetInterfaceID");
  *reinterpret_cast<void**>(&TLGetInterfaceInfo)=getFunction(lp, "TLGetInterfaceInfo");
  *reinterpret_cast<void**>(&TLOpenInterface)=getFunction(lp, "TLOpenInterface");
  *reinterpret_cast<void**>(&TLUpdateInterfaceList)=getFunction(lp, "TLUpdateInterfaceList");
  *reinterpret_cast<void**>(&IFClose)=getFunction(lp, "IFClose");
  *reinterpret_cast<void**>(&IFGetInfo)=getFunction(lp, "IFGetInfo");
  *reinterpret_cast<void**>(&IFGetNumDevices)=getFunction(lp, "IFGetNumDevices");
  *reinterpret_cast<void**>(&IFGetDeviceID)=getFunction(lp, "IFGetDeviceID");
  *reinterpret_cast<void**>(&IFUpdateDeviceList)=getFunction(lp, "IFUpdateDeviceList");
  *reinterpret_cast<void**>(&IFGetDeviceInfo)=getFunction(lp, "IFGetDeviceInfo");
  *reinterpret_cast<void**>(&IFOpenDevice)=getFunction(lp, "IFOpenDevice");

  *reinterpret_cast<void**>(&DevGetPort)=getFunction(lp, "DevGetPort");
  *reinterpret_cast<void**>(&DevGetNumDataStreams)=getFunction(lp, "DevGetNumDataStreams");
  *reinterpret_cast<void**>(&DevGetDataStreamID)=getFunction(lp, "DevGetDataStreamID");
  *reinterpret_cast<void**>(&DevOpenDataStream)=getFunction(lp, "DevOpenDataStream");
  *reinterpret_cast<void**>(&DevGetInfo)=getFunction(lp, "DevGetInfo");
  *reinterpret_cast<void**>(&DevClose)=getFunction(lp, "DevClose");

  *reinterpret_cast<void**>(&DSAnnounceBuffer)=getFunction(lp, "DSAnnounceBuffer");
  *reinterpret_cast<void**>(&DSAllocAndAnnounceBuffer)=getFunction(lp, "DSAllocAndAnnounceBuffer");
  *reinterpret_cast<void**>(&DSFlushQueue)=getFunction(lp, "DSFlushQueue");
  *reinterpret_cast<void**>(&DSStartAcquisition)=getFunction(lp, "DSStartAcquisition");
  *reinterpret_cast<void**>(&DSStopAcquisition)=getFunction(lp, "DSStopAcquisition");
  *reinterpret_cast<void**>(&DSGetInfo)=getFunction(lp, "DSGetInfo");
  *reinterpret_cast<void**>(&DSGetBufferID)=getFunction(lp, "DSGetBufferID");
  *reinterpret_cast<void**>(&DSClose)=getFunction(lp, "DSClose");
  *reinterpret_cast<void**>(&DSRevokeBuffer)=getFunction(lp, "DSRevokeBuffer");
  *reinterpret_cast<void**>(&DSQueueBuffer)=getFunction(lp, "DSQueueBuffer");
  *reinterpret_cast<void**>(&DSGetBufferInfo)=getFunction(lp, "DSGetBufferInfo");

  // Get GenTL version
  uint32_t VerMaj = 1, VerMin=1;
  try
  {
      // Anything possible here
      if(GenTL::GC_ERR_SUCCESS == GCInitLib())
      {
          size_t buff_sz = sizeof(VerMaj);
          GCGetInfo(GenTL::TL_INFO_GENTL_VER_MAJOR, nullptr, &VerMaj, &buff_sz);
          GCGetInfo(GenTL::TL_INFO_GENTL_VER_MINOR, nullptr, &VerMin, &buff_sz);
          GCCloseLib();
      }
  }
  catch (...)
  {
      // Catch all exceptions and then behave as it is GenTL 1.1
      VerMaj = 1;
      VerMin = 1;
  }


  // GenTL v1.1
  if(VerMaj==1 && VerMin>=1)
  {
      *reinterpret_cast<void**>(&GCGetNumPortURLs)=getFunction(lp, "GCGetNumPortURLs");
      *reinterpret_cast<void**>(&GCGetPortURLInfo)=getFunction(lp, "GCGetPortURLInfo");
      *reinterpret_cast<void**>(&GCReadPortStacked)=getFunction(lp, "GCReadPortStacked");
      *reinterpret_cast<void**>(&GCWritePortStacked)=getFunction(lp, "GCWritePortStacked");
  }


  // GenTL v1.3
  if(VerMaj==1 && VerMin>=3)
  {
       *reinterpret_cast<void**>(&DSGetBufferChunkData)=getFunction(lp, "DSGetBufferChunkData");
  }

  // GenTL v1.4
  if(VerMaj==1 && VerMin>=4)
  {
     *reinterpret_cast<void**>(&IFGetParentTL)=getFunction(lp, "IFGetParentTL");
     *reinterpret_cast<void**>(&DevGetParentIF)=getFunction(lp, "DevGetParentIF");
     *reinterpret_cast<void**>(&DSGetParentDev)=getFunction(lp, "DSGetParentDev");
  }

  // GenTL v1.5
  if(VerMaj==1 && VerMin>=5)
  {
    *reinterpret_cast<void**>(&DSGetNumBufferParts)=getFunction(lp, "DSGetNumBufferParts");
    *reinterpret_cast<void**>(&DSGetBufferPartInfo)=getFunction(lp, "DSGetBufferPartInfo");
  }

  lib=static_cast<void *>(lp);
}

GenTLWrapper::~GenTLWrapper()
{
  FreeLibrary(static_cast<HMODULE>(lib));
}

}
