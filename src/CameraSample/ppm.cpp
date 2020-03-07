/*
Copyright 2012-2014 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "ppm.h"
#include "helper_image/helper_common.h"
#include <cctype>
#include <fstream>
#include <cstring>
#include <cmath>

int loadPPM(const char *file, void** data, BaseAllocator *alloc, unsigned int &width, unsigned &wPitch, unsigned int &height, unsigned &bitsPerPixel, unsigned &channels) {
    FILE *fp = nullptr;

	if(FOPEN_FAIL(FOPEN(fp, file, "rb")))
		return 0;

	unsigned int startPosition = 0;

	// check header
	char header[PGMHeaderSize] = { 0 };

	while (header[startPosition] == 0) {
		startPosition = 0;
        if(fgets(header, PGMHeaderSize, fp) == nullptr)
			return 0;

		while (isspace(header[startPosition])) startPosition++;
	}

	bool fvHeader = false;
	int strOffset = 2;
	if(strncmp(&header[startPosition], "P5", 2) == 0) {
		channels = 1;
	}
	else if(strncmp(&header[startPosition], "P6", 2) == 0) {
		channels = 3;
	}
	else if(strncmp(&header[startPosition], "P15", 3) == 0) {     //fv
		channels = 1;
		strOffset = 3;
		fvHeader = true;
	}
	else if(strncmp(&header[startPosition], "P16", 3) == 0) {    //fv
		channels = 3;
		strOffset = 3;
		fvHeader = true;
	}
	else {
		channels = 0;
		return 1;
	}

	// parse header, read maxval, width and height
	unsigned int maxval = 0;
    unsigned int i = 0;
	int readsCount = 0;

    if((i = SSCANF(&header[startPosition + strOffset], "%u %u %u", &width, &height, &maxval)) == (unsigned int)(EOF))
        i = 0;

	while (i < 3) {
        if(fgets(header, PGMHeaderSize, fp) == nullptr)
			return 0;

		if(header[0] == '#')
			continue;

		if(i == 0) {
            if((readsCount = SSCANF(header, "%u %u %u", &width, &height, &maxval)) != EOF)
				i += readsCount;
		}
		else if(i == 1) {
            if((readsCount = SSCANF(header, "%u %u", &height, &maxval)) != EOF)
				i += readsCount;
		}
		else if(i == 2) {
            if((readsCount = SSCANF(header, "%u", &maxval)) != EOF)
				i += readsCount;
		}
	}
	bitsPerPixel = int(log(maxval + 1) / log(2));

	const int bytePerPixel = _uSnapUp<unsigned>(bitsPerPixel, 8) / 8;

	wPitch = channels * _uSnapUp<unsigned>(width * bytePerPixel, alloc->getAlignment());

	*data = alloc->allocate(wPitch * height);
    auto *d = static_cast<unsigned char *>(*data);

	for(unsigned i = 0; i < height; i++) {

		if(fread(&d[i * wPitch], sizeof(unsigned char), width * bytePerPixel * channels, fp) == 0)
			return 0;

		if(bytePerPixel == 2 && !fvHeader)
		{
            auto *p = reinterpret_cast<unsigned short*>(&d[i * wPitch]);
			for(unsigned int x = 0; x < wPitch / bytePerPixel; x++)
			{
				unsigned short t = p[x];
				unsigned short t1 = t >> 8;
				t = (t << 8) | t1;
				p[x] = t;
			}
		}
	}

	fclose(fp);
	return 1;
}

int getFileParameters(const char *file, unsigned &width, unsigned &height) {
    FILE *fp = nullptr;

	if(FOPEN_FAIL(FOPEN(fp, file, "rb")))
		return 0;

	unsigned int startPosition = 0;
    unsigned channels = 0;
	// check header
	char header[PGMHeaderSize] = { 0 };

	while (header[startPosition] == 0) {
		startPosition = 0;
        if(fgets(header, PGMHeaderSize, fp) == nullptr)
			return 0;

		while (isspace(header[startPosition])) startPosition++;
	}

	if(strncmp(&header[startPosition], "P5", 2) == 0) {
//		channels = 1;
	}
	else if(strncmp(&header[startPosition], "P6", 2) == 0) {
//		channels = 3;
	}
	else {
//		channels = 0;
		return 0;
	}

	// parse header, read maxval, width and height
	unsigned int maxval = 0;
    unsigned int i = 0;
    unsigned int readsCount = 0;

    if((i = SSCANF(&header[startPosition + 2], "%u %u %u", &width, &height, &maxval)) == (unsigned int)(EOF))
        i = 0;

	while (i < 3) {
        if(fgets(header, PGMHeaderSize, fp) == nullptr)
			return 0;

		if(header[0] == '#')
			continue;

		if(i == 0) {
            if((readsCount = SSCANF(header, "%u %u %u", &width, &height, &maxval)) != (unsigned int)(EOF))
				i += readsCount;
		}
		else if(i == 1) {
            if((readsCount = SSCANF(header, "%u %u", &height, &maxval)) != (unsigned int)(EOF))
				i += readsCount;
		}
		else if(i == 2) {
            if((readsCount = SSCANF(header, "%u", &maxval)) != (unsigned int)(EOF))
				i += readsCount;
		}
	}

	fclose(fp);
    (void) channels;
	return 1;
}


int savePPM(const char *file, unsigned char *data, unsigned w, unsigned wPitch, unsigned h, int bitsPerPixel, unsigned int channels) {
    assert(nullptr != data);
	assert(w > 0);
	assert(h > 0);

	std::fstream fh(file, std::fstream::out | std::fstream::binary);
	if(fh.bad())
		return 0;

	if(channels == 1) {
		fh << "P5\n";
	}
	else if(channels == 3) {
		fh << "P6\n";
	}
	else
		return 0;

	fh << w << "\n" << h << "\n" << ((1 << bitsPerPixel) - 1) << std::endl;
	const int bytePerPixel = _uSnapUp<unsigned>(bitsPerPixel, 8) / 8;

	for(unsigned int y = 0; y < h && fh.good(); y++)
	{
        if(bytePerPixel == 2)
        {
            auto *p = reinterpret_cast<unsigned short*>(data);
            for(unsigned int x = 0; x < wPitch / bytePerPixel; x++)
            {
                unsigned short t = p[y * wPitch / bytePerPixel + x];
                unsigned short t1 = t >> 8;
                t = (t << 8) | t1;
                p[y * wPitch / bytePerPixel + x] = t;
            }
        }

		fh.write(reinterpret_cast<const char *>(&data[y * wPitch]), w * channels * bytePerPixel);
	}

	fh.flush();
	if(fh.bad())
		return 0;

	fh.close();
	return 1;
}
