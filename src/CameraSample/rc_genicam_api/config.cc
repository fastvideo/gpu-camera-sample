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

#include "config.h"
#include "buffer.h"

#include <stdexcept>
#include <iomanip>

#include "Base/GCException.h"

#include <GenApi/ChunkAdapterGEV.h>
#include <GenApi/ChunkAdapterU3V.h>
#include <GenApi/ChunkAdapterGeneric.h>

#include <rc_genicam_api/pixel_formats.h>

namespace rcg
{

bool callCommand(const std::shared_ptr<GenApi::CNodeMapRef> &nodemap, const char *name,
                 bool exception)
{
  bool ret=false;

  try
  {
    GenApi::INode *node=nodemap->_GetNode(name);

    if (node != 0)
    {
      if (GenApi::IsWritable(node))
      {
        GenApi::ICommand *val=dynamic_cast<GenApi::ICommand *>(node);

        if (val != 0)
        {
          val->Execute();
          ret=true;
        }
        else if (exception)
        {
          throw std::invalid_argument(std::string("Feature not a command: ")+name);
        }
      }
      else if (exception)
      {
        throw std::invalid_argument(std::string("Feature not writable: ")+name);
      }
    }
    else if (exception)
    {
      throw std::invalid_argument(std::string("Feature not found: ")+name);
    }
  }
  catch (const GENICAM_NAMESPACE::GenericException &ex)
  {
    if (exception)
    {
      throw std::invalid_argument(ex.what());
    }
  }

  return ret;
}

bool setBoolean(const std::shared_ptr<GenApi::CNodeMapRef> &nodemap, const char *name,
                bool value, bool exception)
{
  bool ret=false;

  try
  {
    GenApi::INode *node=nodemap->_GetNode(name);

    if (node != 0)
    {
      if (GenApi::IsWritable(node))
      {
        GenApi::IBoolean *val=dynamic_cast<GenApi::IBoolean *>(node);

        if (val != 0)
        {
          val->SetValue(value);
          ret=true;
        }
        else if (exception)
        {
          throw std::invalid_argument(std::string("Feature not boolean: ")+name);
        }
      }
      else if (exception)
      {
        throw std::invalid_argument(std::string("Feature not writable: ")+name);
      }
    }
    else if (exception)
    {
      throw std::invalid_argument(std::string("Feature not found: ")+name);
    }
  }
  catch (const GENICAM_NAMESPACE::GenericException &ex)
  {
    if (exception)
    {
      throw std::invalid_argument(ex.what());
    }
  }

  return ret;
}

bool setInteger(const std::shared_ptr<GenApi::CNodeMapRef> &nodemap, const char *name,
                int64_t value, bool exception)
{
  bool ret=false;

  try
  {
    GenApi::INode *node=nodemap->_GetNode(name);

    if (node != 0)
    {
      if (GenApi::IsWritable(node))
      {
        GenApi::IInteger *val=dynamic_cast<GenApi::IInteger *>(node);

        if (val != 0)
        {
          val->SetValue(value);
          ret=true;
        }
        else if (exception)
        {
          throw std::invalid_argument(std::string("Feature not integer: ")+name);
        }
      }
      else if (exception)
      {
        throw std::invalid_argument(std::string("Feature not writable: ")+name);
      }
    }
    else if (exception)
    {
      throw std::invalid_argument(std::string("Feature not found: ")+name);
    }
  }
  catch (const GENICAM_NAMESPACE::GenericException &ex)
  {
    if (exception)
    {
      throw std::invalid_argument(ex.what());
    }
  }

  return ret;
}

bool setIPV4Address(const std::shared_ptr<GenApi::CNodeMapRef> &nodemap, const char *name,
                    const char *value, bool exception)
{
  bool ret=false;

  try
  {
    GenApi::INode *node=nodemap->_GetNode(name);

    if (node != 0)
    {
      if (GenApi::IsWritable(node))
      {
        GenApi::IInteger *val=dynamic_cast<GenApi::IInteger *>(node);

        if (val != 0)
        {
          int64_t ip=0;

          std::stringstream in(value);
          std::string elem;

          for (int i=0; i<4; i++)
          {
            getline(in, elem, '.');
            ip=(ip<<8)|(stoi(elem)&0xff);
          }

          val->SetValue(ip);
          ret=true;
        }
        else if (exception)
        {
          throw std::invalid_argument(std::string("Feature not integer: ")+name);
        }
      }
      else if (exception)
      {
        throw std::invalid_argument(std::string("Feature not writable: ")+name);
      }
    }
    else if (exception)
    {
      throw std::invalid_argument(std::string("Feature not found: ")+name);
    }
  }
  catch (const GENICAM_NAMESPACE::GenericException &ex)
  {
    if (exception)
    {
      throw std::invalid_argument(ex.what());
    }
  }

  return ret;
}

bool setFloat(const std::shared_ptr<GenApi::CNodeMapRef> &nodemap, const char *name,
              double value, bool exception)
{
  bool ret=false;

  try
  {
    GenApi::INode *node=nodemap->_GetNode(name);

    if (node != 0)
    {
      if (GenApi::IsWritable(node))
      {
        GenApi::IFloat *val=dynamic_cast<GenApi::IFloat *>(node);

        if (val != 0)
        {
          val->SetValue(value);
          ret=true;
        }
        else if (exception)
        {
          throw std::invalid_argument(std::string("Feature not float: ")+name);
        }
      }
      else if (exception)
      {
        throw std::invalid_argument(std::string("Feature not writable: ")+name);
      }
    }
    else if (exception)
    {
      throw std::invalid_argument(std::string("Feature not found: ")+name);
    }
  }
  catch (const GENICAM_NAMESPACE::GenericException &ex)
  {
    if (exception)
    {
      throw std::invalid_argument(ex.what());
    }
  }

  return ret;
}

bool setEnum(const std::shared_ptr<GenApi::CNodeMapRef> &nodemap, const char *name,
             const char *value, bool exception)
{
  bool ret=false;

  try
  {
    GenApi::INode *node=nodemap->_GetNode(name);

    if (node != 0)
    {
      if (GenApi::IsWritable(node))
      {
        GenApi::IEnumeration *val=dynamic_cast<GenApi::IEnumeration *>(node);

        if (val != 0)
        {
          GenApi::IEnumEntry *entry=val->GetEntryByName(value);

          if (entry != 0)
          {
            val->SetIntValue(entry->GetValue());

            return true;
          }
          else if (exception)
          {
            throw std::invalid_argument(std::string("Enumeration '")+name+
                                        "' does not contain: "+value);
          }
        }
        else if (exception)
        {
          throw std::invalid_argument(std::string("Feature not enumeration: ")+name);
        }
      }
      else if (exception)
      {
        throw std::invalid_argument(std::string("Feature not writable: ")+name);
      }
    }
    else if (exception)
    {
      throw std::invalid_argument(std::string("Feature not found: ")+name);
    }
  }
  catch (const GENICAM_NAMESPACE::GenericException &ex)
  {
    if (exception)
    {
      throw std::invalid_argument(ex.what());
    }
  }

  return ret;
}

bool setString(const std::shared_ptr<GenApi::CNodeMapRef> &nodemap, const char *name,
               const char *value, bool exception)
{
  bool ret=false;

  try
  {
    GenApi::INode *node=nodemap->_GetNode(name);

    if (node != 0)
    {
      if (GenApi::IsWritable(node))
      {
        switch (node->GetPrincipalInterfaceType())
        {
          case GenApi::intfIBoolean:
            {
              GenApi::IBoolean *p=dynamic_cast<GenApi::IBoolean *>(node);

              std::string v=std::string(value);
              if (v == "true" || v == "True" || v == "TRUE")
              {
                p->SetValue(1);
              }
              else if (v == "false" || v == "False" || v == "FALSE")
              {
                p->SetValue(0);
              }
              else
              {
				p->SetValue(std::stoi(v)? 1 : 0);
              }
            }
            break;

          case GenApi::intfIInteger:
            {
              GenApi::IInteger *p=dynamic_cast<GenApi::IInteger *>(node);

              switch (p->GetRepresentation())
              {
                case GenApi::HexNumber:
                  p->SetValue(std::stoll(std::string(value), 0, 16));
                  break;

                case GenApi::IPV4Address:
                  {
                    int64_t ip=0;

                    std::stringstream in(value);
                    std::string elem;

                    for (int i=0; i<4; i++)
                    {
                      getline(in, elem, '.');
                      ip=(ip<<8)|(stoi(elem)&0xff);
                    }

                    p->SetValue(ip);
                  }
                  break;

                case GenApi::MACAddress:
                  {
                    int64_t mac=0;

                    std::stringstream in(value);
                    std::string elem;

                    for (int i=0; i<4; i++)
                    {
                      getline(in, elem, ':');
                      mac=(mac<<8)|(stoi(elem, 0, 16)&0xff);
                    }

                    p->SetValue(mac);
                  }
                  break;

                default:
                  p->SetValue(std::stoll(std::string(value)));
                  break;
              }
            }
            break;

          case GenApi::intfIFloat:
            {
              GenApi::IFloat *p=dynamic_cast<GenApi::IFloat *>(node);
              p->SetValue(std::stof(std::string(value)));
            }
            break;

          case GenApi::intfIEnumeration:
            {
              GenApi::IEnumeration *p=dynamic_cast<GenApi::IEnumeration *>(node);
              GenApi::IEnumEntry *entry=p->GetEntryByName(value);

              if (entry != 0)
              {
                p->SetIntValue(entry->GetValue());
              }
              else if (exception)
              {
                throw std::invalid_argument(std::string("Enumeration '")+name+
                                            "' does not contain: "+value);
              }
            }
            break;

          case GenApi::intfIString:
            {
              GenApi::IString *p=dynamic_cast<GenApi::IString *>(node);
              p->SetValue(value);
            }
            break;

          default:
            if (exception)
            {
              throw std::invalid_argument(std::string("Feature of unknown datatype: ")+name);
            }
            break;
        }
      }
      else if (exception)
      {
        throw std::invalid_argument(std::string("Feature not writable: ")+name);
      }
    }
    else if (exception)
    {
      throw std::invalid_argument(std::string("Feature not found: ")+name);
    }
  }
  catch (const GENICAM_NAMESPACE::GenericException &ex)
  {
    if (exception)
    {
      throw std::invalid_argument(ex.what());
    }
  }

  return ret;
}

bool getBoolean(const std::shared_ptr<GenApi::CNodeMapRef> &nodemap, const char *name,
                bool exception, bool igncache)
{
  bool ret=false;

  try
  {
    GenApi::INode *node=nodemap->_GetNode(name);

    if (node != 0)
    {
      if (GenApi::IsReadable(node))
      {
        GenApi::IBoolean *val=dynamic_cast<GenApi::IBoolean *>(node);

        if (val != 0)
        {
          ret=val->GetValue(false, igncache);
        }
        else if (exception)
        {
          throw std::invalid_argument(std::string("Feature not boolean: ")+name);
        }
      }
      else if (exception)
      {
        throw std::invalid_argument(std::string("Feature not readable: ")+name);
      }
    }
    else if (exception)
    {
      throw std::invalid_argument(std::string("Feature not found: ")+name);
    }
  }
  catch (const GENICAM_NAMESPACE::GenericException &ex)
  {
    if (exception)
    {
      throw std::invalid_argument(ex.what());
    }
  }

  return ret;
}

int64_t getInteger(const std::shared_ptr<GenApi::CNodeMapRef> &nodemap, const char *name,
                   int64_t *vmin, int64_t *vmax, bool exception, bool igncache)
{
  int64_t ret=0;

  if (vmin != 0) *vmin=0;
  if (vmax != 0) *vmax=0;

  try
  {
    GenApi::INode *node=nodemap->_GetNode(name);

    if (node != 0)
    {
      if (GenApi::IsReadable(node))
      {
        GenApi::IInteger *val=dynamic_cast<GenApi::IInteger *>(node);

        if (val != 0)
        {
          ret=val->GetValue(false, igncache);

          if (vmin != 0) *vmin=val->GetMin();
          if (vmax != 0) *vmax=val->GetMax();
        }
        else if (exception)
        {
          throw std::invalid_argument(std::string("Feature not integer: ")+name);
        }
      }
      else if (exception)
      {
        throw std::invalid_argument(std::string("Feature not readable: ")+name);
      }
    }
    else if (exception)
    {
      throw std::invalid_argument(std::string("Feature not found: ")+name);
    }
  }
  catch (const GENICAM_NAMESPACE::GenericException &ex)
  {
    if (exception)
    {
      throw std::invalid_argument(ex.what());
    }
  }

  return ret;
}

double getFloat(const std::shared_ptr<GenApi::CNodeMapRef> &nodemap, const char *name,
                double *vmin, double *vmax, bool exception, bool igncache)
{
  double ret=0;

  if (vmin != 0) *vmin=0;
  if (vmax != 0) *vmax=0;

  try
  {
    GenApi::INode *node=nodemap->_GetNode(name);

    if (node != 0)
    {
      if (GenApi::IsReadable(node))
      {
        GenApi::IFloat *val=dynamic_cast<GenApi::IFloat *>(node);

        if (val != 0)
        {
          ret=val->GetValue(false, igncache);

          if (vmin != 0) *vmin=val->GetMin();
          if (vmax != 0) *vmax=val->GetMax();
        }
        else if (exception)
        {
          throw std::invalid_argument(std::string("Feature not float: ")+name);
        }
      }
      else if (exception)
      {
        throw std::invalid_argument(std::string("Feature not readable: ")+name);
      }
    }
    else if (exception)
    {
      throw std::invalid_argument(std::string("Feature not found: ")+name);
    }
  }
  catch (const GENICAM_NAMESPACE::GenericException &ex)
  {
    if (exception)
    {
      throw std::invalid_argument(ex.what());
    }
  }

  return ret;
}

std::string getEnum(const std::shared_ptr<GenApi::CNodeMapRef> &nodemap, const char *name,
                    bool exception)
{
  std::string ret;

  try
  {
    GenApi::INode *node=nodemap->_GetNode(name);

    if (node != 0)
    {
      if (GenApi::IsReadable(node))
      {
        GenApi::IEnumeration *val=dynamic_cast<GenApi::IEnumeration *>(node);

        if (val != 0)
        {
          GenApi::IEnumEntry *entry=val->GetCurrentEntry();

          if (entry != 0)
          {
            ret=entry->GetSymbolic();
          }
          else if (exception)
          {
            throw std::invalid_argument(std::string("Current value is not defined: ")+name);
          }
        }
        else if (exception)
        {
          throw std::invalid_argument(std::string("Feature not enumeration: ")+name);
        }
      }
      else if (exception)
      {
        throw std::invalid_argument(std::string("Feature not readable: ")+name);
      }
    }
    else if (exception)
    {
      throw std::invalid_argument(std::string("Feature not found: ")+name);
    }
  }
  catch (const GENICAM_NAMESPACE::GenericException &ex)
  {
    if (exception)
    {
      throw std::invalid_argument(ex.what());
    }
  }

  return ret;
}

std::string getEnum(const std::shared_ptr<GenApi::CNodeMapRef> &nodemap, const char *name,
                    std::vector<std::string> &list, bool exception)
{
  std::string ret;

  list.clear();

  try
  {
    GenApi::INode *node=nodemap->_GetNode(name);

    if (node != 0)
    {
      if (GenApi::IsReadable(node))
      {
        GenApi::IEnumeration *val=dynamic_cast<GenApi::IEnumeration *>(node);

        if (val != 0)
        {
          ret=val->GetCurrentEntry()->GetSymbolic();

          GenApi::StringList_t entry;
          val->GetSymbolics(entry);

          for (size_t i=0; i<entry.size(); i++)
          {
            list.push_back(std::string(entry[i]));
          }
        }
        else if (exception)
        {
          throw std::invalid_argument(std::string("Feature not enumeration: ")+name);
        }
      }
      else if (exception)
      {
        throw std::invalid_argument(std::string("Feature not readable: ")+name);
      }
    }
    else if (exception)
    {
      throw std::invalid_argument(std::string("Feature not found: ")+name);
    }
  }
  catch (const GENICAM_NAMESPACE::GenericException &ex)
  {
    if (exception)
    {
      throw std::invalid_argument(ex.what());
    }
  }

  return ret;
}

std::string getString(const std::shared_ptr<GenApi::CNodeMapRef> &nodemap, const char *name,
                      bool exception, bool igncache)
{
  std::ostringstream out;

  try
  {
    GenApi::INode *node=nodemap->_GetNode(name);

    if (node != 0)
    {
      if (GenApi::IsReadable(node))
      {
        switch (node->GetPrincipalInterfaceType())
        {
          case GenApi::intfIBoolean:
            {
              GenApi::IBoolean *p=dynamic_cast<GenApi::IBoolean *>(node);
              out << p->GetValue(false, igncache);
            }
            break;

          case GenApi::intfIInteger:
            {
              GenApi::IInteger *p=dynamic_cast<GenApi::IInteger *>(node);
              int64_t value=p->GetValue(false, igncache);

              switch (p->GetRepresentation())
              {
                case GenApi::HexNumber:
                  out << std::hex << value;
                  break;

                case GenApi::IPV4Address:
                   out << ((value>>24)&0xff) << '.' << ((value>>16)&0xff) << '.'
                       << ((value>>8)&0xff) << '.' << (value&0xff);
                   break;

                case GenApi::MACAddress:
                   out << std::hex << std::setfill('0');
                   out << std::setw(2) << ((value>>40)&0xff) << ':'
                       << std::setw(2) << ((value>>32)&0xff) << ':'
                       << std::setw(2) << ((value>>24)&0xff) << ':'
                       << std::setw(2) << ((value>>16)&0xff) << ':'
                       << std::setw(2) << ((value>>8)&0xff) << ':'
                       << std::setw(2) << (value&0xff);
                   break;

                default:
                  out << value;
                  break;
              }
            }
            break;

          case GenApi::intfIFloat:
            {
              GenApi::IFloat *p=dynamic_cast<GenApi::IFloat *>(node);
              out << p->GetValue(false, igncache);
            }
            break;

          case GenApi::intfIEnumeration:
            {
              GenApi::IEnumeration *p=dynamic_cast<GenApi::IEnumeration *>(node);
              out << p->GetCurrentEntry()->GetSymbolic();
            }
            break;

          case GenApi::intfIString:
            {
              GenApi::IString *p=dynamic_cast<GenApi::IString *>(node);
              out << p->GetValue(false, igncache);
            }
            break;

          default:
            if (exception)
            {
              throw std::invalid_argument(std::string("Feature of unknown datatype: ")+name);
            }
            break;
        }
      }
      else if (exception)
      {
        throw std::invalid_argument(std::string("Feature not readable: ")+name);
      }
    }
    else if (exception)
    {
      throw std::invalid_argument(std::string("Feature not found: ")+name);
    }
  }
  catch (const GENICAM_NAMESPACE::GenericException &ex)
  {
    if (exception)
    {
      throw std::invalid_argument(ex.what());
    }
  }

  return out.str();
}

void checkFeature(const std::shared_ptr<GenApi::CNodeMapRef> &nodemap, const char *name,
                  const char *value, bool igncache)
{
  std::string cvalue=rcg::getString(nodemap, name, true, igncache);

  if (cvalue != "" && cvalue != value)
  {
    std::ostringstream out;
    out << name << " == " << value << " expected: " << cvalue;
    throw std::invalid_argument(out.str());
  }
}

std::shared_ptr<GenApi::CChunkAdapter> getChunkAdapter(const std::shared_ptr<GenApi::CNodeMapRef> &nodemap,
                                                       const std::string &tltype)
{
  std::shared_ptr<GenApi::CChunkAdapter> chunkadapter;

  if (setBoolean(nodemap, "ChunkModeActive", true))
  {
    if (tltype == "GEV")
    {
      chunkadapter=std::shared_ptr<GenApi::CChunkAdapter>(new GenApi::CChunkAdapterGEV(nodemap->_Ptr));
    }
    else if (tltype == "U3V")
    {
      chunkadapter=std::shared_ptr<GenApi::CChunkAdapter>(new GenApi::CChunkAdapterU3V(nodemap->_Ptr));
    }
    else
    {
      chunkadapter=std::shared_ptr<GenApi::CChunkAdapter>(new GenApi::CChunkAdapterGeneric(nodemap->_Ptr));
    }
  }

  return chunkadapter;
}

std::string getComponetOfPart(const std::shared_ptr<GenApi::CNodeMapRef> &nodemap,
                              const rcg::Buffer *buffer, uint32_t ipart)
{
  std::string component;

  try
  {
    // get chunk component selector and proprietary chunk part index parmeters

    GenApi::IEnumeration *sel=dynamic_cast<GenApi::IEnumeration *>(nodemap->_GetNode("ChunkComponentSelector"));
    GenApi::IInteger *part=dynamic_cast<GenApi::IInteger *>(nodemap->_GetNode("ChunkPartIndex"));

    if (sel != 0 && part != 0)
    {
      if (GenApi::IsReadable(sel) && GenApi::IsWritable(sel) && GenApi::IsReadable(part))
      {
        // go through all available enumerations

        GenApi::NodeList_t list;
        sel->GetEntries(list);

        for (size_t i=0; i<list.size() && component.size() == 0; i++)
        {
          GenApi::IEnumEntry *entry=dynamic_cast<GenApi::IEnumEntry *>(list[i]);

          if (entry != 0 && GenApi::IsReadable(entry))
          {
            sel->SetIntValue(entry->GetValue());

            int64_t val=part->GetValue();
            if (val == ipart)
            {
              component=dynamic_cast<GenApi::IEnumEntry *>(list[i])->GetSymbolic();
            }
          }
        }
      }
    }
  }
  catch (const std::exception &)
  { /* ignore errors */ }
  catch (const GENICAM_NAMESPACE::GenericException &)
  { /* ignore errors */ }

  // try guessing component name from pixel format

  if (component.size() == 0 && buffer->getImagePresent(ipart))
  {
    switch (buffer->getPixelFormat(ipart))
    {
      case Mono8:
      case YCbCr411_8:
        if (buffer->getWidth(ipart) >= buffer->getHeight(ipart))
        {
          component="Intensity";
        }
        else
        {
          component="IntensityCombined";
        }
        break;

      case Coord3D_C16:
        component="Disparity";
        break;

      case Confidence8:
        component="Confidence";
        break;

      case Error8:
        component="Error";
        break;

      default:
        break;
    }
  }

  return component;
}

}
