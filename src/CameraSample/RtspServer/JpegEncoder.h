#ifndef JPEG_ENCODER_H
#define JPEG_ENCODER_H

#include <QImage>
#include <QByteArray>
#include <QSharedPointer>

#include "common_utils.h"

class jpeg_encoder
{
public:
	jpeg_encoder();
	~jpeg_encoder();

    bool encode(unsigned char* input, int width, int height, int channels, std::vector<uchar>& output, int quality = 60);
    bool encode(unsigned char* input, int width, int height, int channels, uchar* output, uint &size, int quality = 60);

private:

};

#endif // JPEG_ENCODER_H
