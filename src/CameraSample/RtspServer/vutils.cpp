#include "vutils.h"

#include "JpegEncoder.h"

void encodeJpeg(int idthread, unsigned char* data, int width, int height, int channels, bytearray& output)
{
#if 0
	QImage::Format fmt = QImage::Format_Grayscale8;
	if(channels == 3)
		fmt = QImage::Format_RGB888;
	QImage img(data, width, height, fmt);

	QByteArray d;
	QDataStream stream(&d, QIODevice::WriteOnly);

	QImageWriter writer(stream.device(), "jpeg");
	writer.setQuality(40);

	writer.write(img);
	output.resize(d.size());
	std::copy(d.data(), d.data() + d.size(), output.data());
#else
    idthread;
	jpeg_encoder enc;
    enc.encode(data, width, height, channels, output, 30);
#endif
}

void copyPartImage(unsigned char *input, size_t xoff, size_t yoff, size_t sizeEl, size_t linesize,
				   size_t hpart, size_t linesizepart, unsigned char *output)
{
	for(size_t y = 0; y < hpart; ++y){
		unsigned char *d = &input[(yoff + y) * linesize + xoff * sizeEl];
		unsigned char *o = &output[y * linesizepart];
		std::copy(d, d + linesizepart, o);
	}
}
