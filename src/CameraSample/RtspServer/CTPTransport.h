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
#ifndef CTPTRANSPORT_H
#define CTPTRANSPORT_H

#include <QDataStream>
#include <QList>

#include "common_utils.h"

const quint32 headerId = 0x01100110;
const quint32 max_packet_data_size = 60000;
const quint32 buffersize_udp = 5000000;

class CTPTransport
{
public:
    CTPTransport();

    void createPacket(const uchar *dataPtr, int len, std::vector<QByteArray> &output);

    QByteArray getPacket();
    quint32 SN() const;

    bool addUdpPacket(const uchar *dataPtr, int len);
    bool isPacketAssembly() const;
    void clearPacket();

private:
    qint32 m_SN = 0;

    struct Udp{
        QByteArray d;
        quint32 off = 0;
        quint32 size = 0;
        quint32 id = 0;
    };
    std::vector<Udp> m_udpPackets;
    QByteArray m_packet;

    void assemplyPacket();

};

#endif // CTPTRANSPORT_H
