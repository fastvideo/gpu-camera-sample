//
// Created by 1 on 28.11.2019.
//

#ifndef WEBCAM_JPEGENC_H
#define WEBCAM_JPEGENC_H

#include "common.h"
#include "common_utils.h"

class jpegenc {
public:
    jpegenc();

    bool encode(const bytearray input[3], int width, int height, bytearray& output, int quality);
    bool decode(const bytearray& input, PImage &image);
    bool decode(const uint8_t* input, int len, PImage &image);

};


#endif //WEBCAM_JPEGENC_H
