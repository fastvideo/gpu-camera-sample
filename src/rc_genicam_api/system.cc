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

#include "system.h"

#include "gentl_wrapper.h"
#include "exception.h"
#include "interface.h"
#include "cport.h"

#include <iostream>

#ifdef _WIN32
#include <Windows.h>
#include <cstring>
#undef min
#undef max
#endif

namespace rcg
{

System::~System()
{
  if (n_open > 0 && tl != 0)
  {
    gentl->TLClose(tl);
  }

  gentl->GCCloseLib();
}

namespace
{

std::mutex system_mtx;
std::vector<std::shared_ptr<System> > slist;

int find(const std::vector<std::shared_ptr<System> > &list, const std::string &filename)
{
  for (size_t i=0; i<list.size(); i++)
  {
    if (list[i]->getFilename() == filename)
    {
      return static_cast<int>(i);
    }
  }

  return -1;
}

#ifdef _WIN32
static std::string getPathToThisDll()
{
  HMODULE hm = nullptr;
  if (GetModuleHandleEx(
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    reinterpret_cast<LPCSTR>(&getPathToThisDll), &hm) == 0)
  {
    return {};
  }

  char path[MAX_PATH];
  if (GetModuleFileName(hm, path, sizeof(path)) == 0)
  {
    return {};
  }

  std::string p{ path };
  const auto bs_pos = p.rfind('\\');
  if (bs_pos != std::string::npos)
  {
    p = p.substr(0, bs_pos);
  }

  return p;
}

#endif

}

std::vector<std::shared_ptr<System> > System::getSystems()
{
  std::lock_guard<std::mutex> lock(system_mtx);
  std::vector<std::shared_ptr<System> > ret;

  // get list of all available transport layer libraries

  const char *env=0;
  if (sizeof(size_t) == 8)
  {
    env="GENICAM_GENTL64_PATH";
  }
  else
  {
    env="GENICAM_GENTL32_PATH";
  }

  std::string path;

  const char *envpath=std::getenv(env);
  if (envpath != 0)
  {
    path=envpath;
  }

  if (path.size() == 0)
  {
#ifdef _WIN32
    // under Windows, use the path to the current executable as fallback

    const size_t n=256;
    char procpath[n];
    std::string path_to_exe;
    if (GetModuleFileName(NULL, procpath, n-1) > 0)
    {
      procpath[n-1]='\0';

      char *p=strrchr(procpath, '\\');
      if (p != 0) *p='\0';

      path_to_exe=procpath;
      path+=";" + path_to_exe;
    }

    const auto path_to_this_dll = getPathToThisDll();
    if (!path_to_this_dll.empty() && path_to_this_dll != path_to_exe)
    {
      path += ";" + path_to_this_dll;
    }
#else
    // otherwise, use the absolute install path to the default transport layer

    path = GENTL_INSTALL_PATH ; // "/opt/XIMEA/lib/";
#endif
  }

  std::vector<std::string> name=getAvailableGenTLs(path.c_str());

  // create list of systems according to the list, using either existing
  // systems or instantiating new ones

  for (size_t i=0; i<name.size(); i++)
  {
    int k=find(slist, name[i]);

    if (k >= 0)
    {
      ret.push_back(slist[static_cast<size_t>(k)]);
    }
    else
    {
      try
      {
        System *p=new System(name[i]);
        ret.push_back(std::shared_ptr<System>(p));
      }
      catch (const std::exception &)
      {
        // ignore transport layers that cannot be used
      }
    }
  }

  // remember returned list for reusing existing systems on the next call

  slist=ret;

  // throw exception if no transport layers are available

  if (ret.size() == 0)
  {
    throw GenTLException(std::string("No transport layers found in path ")+path);
  }

  return ret;
}

void System::clearSystems()
{
  std::lock_guard<std::mutex> lock(system_mtx);
  slist.clear();
}

const std::string &System::getFilename() const
{
  return filename;
}

void System::open()
{
  std::lock_guard<std::mutex> lock(mtx);

  if (n_open == 0)
  {
    if (gentl->TLOpen(&tl) != GenTL::GC_ERR_SUCCESS)
    {
      throw GenTLException("System::open()", gentl);
    }
  }

  n_open++;
}

void System::close()
{
  std::lock_guard<std::mutex> lock(mtx);

  if (n_open > 0)
  {
    n_open--;
  }

  if (n_open == 0)
  {
    gentl->TLClose(tl);
    tl=0;

    nodemap=0;
    cport=0;
  }
}

namespace
{

int find(const std::vector<std::shared_ptr<Interface> > &list, const std::string &id)
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

std::vector<std::shared_ptr<Interface> > System::getInterfaces()
{
  std::lock_guard<std::mutex> lock(mtx);
  std::vector<std::shared_ptr<Interface> > ret;

  if (tl != 0)
  {
    // get list of previously requested interfaces that are still in use

    std::vector<std::shared_ptr<Interface> > current;

    for (size_t i=0; i<ilist.size(); i++)
    {
      std::shared_ptr<Interface> p=ilist[i].lock();
      if (p)
      {
        current.push_back(p);
      }
    }

    // update available interfaces

    if (gentl->TLUpdateInterfaceList(tl, 0, 10) != GenTL::GC_ERR_SUCCESS)
    {
      throw GenTLException("System::getInterfaces()", gentl);
    }

    // create list of interfaces, using either existing interfaces or
    // instantiating new ones

    uint32_t n=0;
    if (gentl->TLGetNumInterfaces(tl, &n) != GenTL::GC_ERR_SUCCESS)
    {
      throw GenTLException("System::getInterfaces()", gentl);
    }

    for (uint32_t i=0; i<n; i++)
    {
      char tmp[256]="";
      size_t size=sizeof(tmp);

      if (gentl->TLGetInterfaceID(tl, i, tmp, &size) != GenTL::GC_ERR_SUCCESS)
      {
        throw GenTLException("System::getInterfaces()", gentl);
      }

      int k=find(current, tmp);

      if (k >= 0)
      {
        ret.push_back(current[static_cast<size_t>(k)]);
      }
      else
      {
        ret.push_back(std::shared_ptr<Interface>(new Interface(shared_from_this(), gentl, tmp)));
      }
    }

    // update internal list of interfaces for reusage on next call

    ilist.clear();
    for (size_t i=0; i<ret.size(); i++)
    {
      ilist.push_back(ret[i]);
    }
  }

  return ret;
}

namespace
{

std::string cTLGetInfo(GenTL::TL_HANDLE tl, const std::shared_ptr<const GenTLWrapper> &gentl,
                       GenTL::TL_INFO_CMD info)
{
  std::string ret;

  GenTL::INFO_DATATYPE type;
  char tmp[1024]="";
  size_t tmp_size=sizeof(tmp);
  GenTL::GC_ERROR err=GenTL::GC_ERR_SUCCESS;

  if (tl != 0)
  {
    err=gentl->TLGetInfo(tl, info, &type, tmp, &tmp_size);
  }
  else
  {
    err=gentl->GCGetInfo(info, &type, tmp, &tmp_size);
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

std::string System::getID()
{
  std::lock_guard<std::mutex> lock(mtx);
  return cTLGetInfo(tl, gentl, GenTL::TL_INFO_ID);
}

std::string System::getVendor()
{
  std::lock_guard<std::mutex> lock(mtx);
  return cTLGetInfo(tl, gentl, GenTL::TL_INFO_VENDOR);
}

std::string System::getModel()
{
  std::lock_guard<std::mutex> lock(mtx);
  return cTLGetInfo(tl, gentl, GenTL::TL_INFO_MODEL);
}

std::string System::getVersion()
{
  std::lock_guard<std::mutex> lock(mtx);
  return cTLGetInfo(tl, gentl, GenTL::TL_INFO_VERSION);
}

std::string System::getTLType()
{
  std::lock_guard<std::mutex> lock(mtx);
  return cTLGetInfo(tl, gentl, GenTL::TL_INFO_TLTYPE);
}

std::string System::getName()
{
  std::lock_guard<std::mutex> lock(mtx);
  return cTLGetInfo(tl, gentl, GenTL::TL_INFO_NAME);
}

std::string System::getPathname()
{
  std::lock_guard<std::mutex> lock(mtx);
  return cTLGetInfo(tl, gentl, GenTL::TL_INFO_PATHNAME);
}

std::string System::getDisplayName()
{
  std::lock_guard<std::mutex> lock(mtx);
  return cTLGetInfo(tl, gentl, GenTL::TL_INFO_DISPLAYNAME);
}

bool System::isCharEncodingASCII()
{
  std::lock_guard<std::mutex> lock(mtx);
  bool ret=true;

  GenTL::INFO_DATATYPE type;
  int32_t v;
  size_t size=sizeof(v);
  GenTL::GC_ERROR err=GenTL::GC_ERR_SUCCESS;

  if (tl != 0)
  {
    err=gentl->TLGetInfo(tl, GenTL::TL_INFO_CHAR_ENCODING, &type, &v, &size);
  }
  else
  {
    err=gentl->GCGetInfo(GenTL::TL_INFO_CHAR_ENCODING, &type, &v, &size);
  }

  if (err == GenTL::GC_ERR_SUCCESS && type == GenTL::INFO_DATATYPE_INT32 &&
      v != GenTL::TL_CHAR_ENCODING_ASCII)
  {
    ret=false;
  }

  return ret;
}

int System::getMajorVersion()
{
  std::lock_guard<std::mutex> lock(mtx);
  uint32_t ret=0;

  GenTL::INFO_DATATYPE type;
  size_t size=sizeof(ret);

  if (tl != 0)
  {
    gentl->TLGetInfo(tl, GenTL::TL_INFO_GENTL_VER_MAJOR, &type, &ret, &size);
  }
  else
  {
    gentl->GCGetInfo(GenTL::TL_INFO_GENTL_VER_MAJOR, &type, &ret, &size);
  }

  return static_cast<int>(ret);
}

int System::getMinorVersion()
{
  std::lock_guard<std::mutex> lock(mtx);
  uint32_t ret=0;

  GenTL::INFO_DATATYPE type;
  size_t size=sizeof(ret);

  if (tl != 0)
  {
    gentl->TLGetInfo(tl, GenTL::TL_INFO_GENTL_VER_MINOR, &type, &ret, &size);
  }
  else
  {
    gentl->GCGetInfo(GenTL::TL_INFO_GENTL_VER_MINOR, &type, &ret, &size);
  }

  return static_cast<int>(ret);
}

std::shared_ptr<GenApi::CNodeMapRef> System::getNodeMap()
{
  std::lock_guard<std::mutex> lock(mtx);
  if (tl != 0 && !nodemap)
  {
    cport=std::shared_ptr<CPort>(new CPort(gentl, &tl));
    nodemap=allocNodeMap(gentl, tl, cport.get());
  }

  return nodemap;
}

void *System::getHandle() const
{
  return tl;
}

System::System(const std::string &_filename)
{
  filename=_filename;

  gentl=std::shared_ptr<const GenTLWrapper>(new GenTLWrapper(filename));

  if (gentl->GCInitLib() != GenTL::GC_ERR_SUCCESS)
  {
    throw GenTLException("System::System()", gentl);
  }

  n_open=0;
  tl=0;
}

}
