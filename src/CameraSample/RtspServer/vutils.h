#ifndef VUTILS_H
#define VUTILS_H

#include "common_utils.h"

void encodeJpeg(int idthread, unsigned char* data, int width, int height, int channels, bytearray& output);

void copyPartImage(unsigned char *input, size_t xoff, size_t yoff, size_t sizeEl, size_t linesize,
				   size_t hpart, size_t linesizepart, unsigned char *output);


#endif // VUTILS_H
