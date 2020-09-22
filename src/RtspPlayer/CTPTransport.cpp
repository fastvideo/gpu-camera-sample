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
        qDebug("ctp: error of begin packet, current count of packets %d", static_cast<int>(m_udpPackets.size()));
        clearPacket();
    }
    if(id > 0 && m_udpPackets.empty()){
        qDebug("ctp: error of continuing");
        return false;
    }

    if(off == 0){
        m_SN = sn;
		m_starttime = getNow();
    }
    if(sn != m_SN){
        qDebug("ctp: error of serial number");
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
        mMutex.lock();
		m_durations["assembly_packet"] = getDuration(m_starttime);
        mMutex.unlock();
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

QMap<QString, double> CTPTransport::durations()
{
    mMutex.lock();
    QMap<QString, double> durs = m_durations;
    mMutex.unlock();
    return durs;
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
            qDebug("ctp: error of order packet. need - %d, current - %d", id + 1, p.id);
            break;
        }
        m_packet.append(p.d);
        id = p.id;
    }
    if(m_packet.size() != needed_size){
        qDebug("ctp: packet did not assembled. needed - %d, current - %d", needed_size, m_packet.size());
        clearPacket();
    }
}
