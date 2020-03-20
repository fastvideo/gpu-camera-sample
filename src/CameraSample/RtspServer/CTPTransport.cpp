/*
 Copyright 2011-2019 Fastvideo, LLC.
 All rights reserved.

 This file is a part of the GPUCameraSample project
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 3. Any third-party SDKs from that project (XIMEA SDK, Fastvideo SDK, etc.) are licensed on different terms. Please see their corresponding license terms.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 The views and conclusions contained in the software and documentation are those
 of the authors and should not be interpreted as representing official policies,
 either expressed or implied, of the FreeBSD Project.
*/

#include "CTPTransport.h"

CTPTransport::CTPTransport()
{

}

void CTPTransport::createPacket(const uchar *dataPtr, int len, std::vector<QByteArray> &output)
{
    output.clear();
    int size = len, off = 0, id = 0;
    char *pos = (char*)(dataPtr);
    while(size > 0){
        QByteArray data;
        QDataStream stream(&data, QIODevice::WriteOnly);

        quint32 l = std::min(max_packet_data_size, static_cast<quint32>(size));

        stream << (quint32)headerId;
        stream << (quint32)m_SN;
        stream << (quint32)id++;
        stream << (quint32)off;
        stream << (quint32)len;
        stream.writeRawData(pos, l);
        size -= l;
        pos += l;
        off += l;

        output.push_back(data);
    }
    m_SN++;
}

QByteArray CTPTransport::getPacket()
{
    return m_packet;
}

quint32 CTPTransport::SN() const
{
    return m_SN;
}

bool CTPTransport::addUdpPacket(const uchar *dataPtr, int len)
{
    QByteArray data((char*)dataPtr, len);
    QDataStream stream(data);

    quint32 header, off, size, sn, id;

    stream >> header;

    if(header != headerId)
        return false;

    stream >> sn;
    stream >> id;
    stream >> off;
    stream >> size;

    if(id == 0 && !m_udpPackets.empty()){
        qDebug("ctp: error of begin packet, current count of packets %d\n", static_cast<int>(m_udpPackets.size()));
        clearPacket();
    }
    if(id > 0 && m_udpPackets.empty()){
        qDebug("ctp: error of continuing \n");
        return false;
    }

    if(off == 0){
        m_SN = sn;
    }
    if(sn != m_SN){
        qDebug("ctp: error of serial number\n");
        clearPacket();
    }

    int l = stream.device()->size() - stream.device()->pos();

    Udp pkt;
    pkt.d.resize(l);
    pkt.size = size;
    pkt.off = off;

    stream.readRawData(pkt.d.data(), l);
    m_udpPackets.push_back(pkt);

    if(off + l == pkt.size){
        assemplyPacket();
    }

    return true;
}

bool CTPTransport::isPacketAssembly() const
{
    return !m_packet.isEmpty();
}

void CTPTransport::clearPacket()
{
    m_udpPackets.clear();
    m_packet.clear();
}

void CTPTransport::assemplyPacket()
{
    m_packet.clear();
    if(m_udpPackets.empty())
        return;

    quint32 id = m_udpPackets.front().id;
    quint32 needed_size = m_udpPackets.front().size;
    for(Udp& p: m_udpPackets){
        if(p.id != id && p.id != id + 1){
            m_packet.clear();
            m_udpPackets.clear();
            qDebug("ctp: error of order packet. need - %d, current - %d\n", id + 1, p.id);
            break;
        }
        m_packet.append(p.d);
        id = p.id;
    }
    if(m_packet.size() != needed_size){
        qDebug("ctp: packet did not assembled. needed - %d, current - %d\n", needed_size, m_packet.size());
        clearPacket();
    }
}
