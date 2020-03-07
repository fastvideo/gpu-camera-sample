/*
Copyright 2012-2018 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __HELPER_JPEG__
#define __HELPER_JPEG__

#include "fastvideo_sdk.h"

#define BYTE_SIZE 8U
#define DCT_SIZE 8U
#define MAX_CODE_LEN 16U

inline unsigned Map_host(unsigned map, unsigned pos) {
    return (map >> (pos * BYTE_SIZE)) & 0xFF;
}

fastStatus_t jfifLoadHeader(
	fastJfifInfo_t *jfifInfo,
	std::istream &fd
);

fastStatus_t jfifLoadHeaderWithoutExif(
    fastJfifInfo_t *jfifInfo,
    std::istream &fd
);

fastStatus_t fastJfifLoadFromFile(const char *filename, fastJfifInfo_t *jfifInfo);
fastStatus_t fastJfifHeaderLoadFromFile(const char *filename, fastJfifInfo_t *jfifInfo);
fastStatus_t fastJfifBytestreamLoadFromFile(const char *filename, fastJfifInfo_t *jfifInfo);

fastStatus_t fastJfifHeaderLoadFromMemory(
	const unsigned char	*inputStream,
	unsigned inputStreamSize,

	fastJfifInfo_t *jfifInfo
);

fastStatus_t fastJfifLoadFromMemory(
	const unsigned char	*inputStream,
	unsigned inputStreamSize,

	fastJfifInfo_t *jfifInfo
);

fastStatus_t fastJfifBytestreamLoadFromMemory(
	const unsigned char	*inputStream,
	unsigned inputStreamSize,

	fastJfifInfo_t *jfifInfo
);

fastStatus_t DLL fastJfifStoreToFile(
	const char *filename,
	fastJfifInfo_t *jfifInfo
);

fastStatus_t DLL fastJfifStoreToMemory(
	unsigned char *outputStream,
	unsigned *outputStreamSize,

	fastJfifInfo_t *jfifInfo
);

fastStatus_t fastJfifLoadFromMemoryNoCopyData(
    const unsigned char	*inputStream,
    unsigned inputStreamSize,

    fastJfifInfo_t *jfifInfo
);

#endif // __HELPER_JPEG__
