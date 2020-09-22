#ifndef CTPTRANSPORT_H
#define CTPTRANSPORT_H

#include <QDataStream>
#include <QList>
#include <QMap>
#include <QMutex>

#include "common.h"
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

	QMap<QString, double> durations();

private:
    qint32 m_SN = 0;

	QMap<QString, double> m_durations;
    timepoint m_starttime;

    QMutex mMutex;

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
