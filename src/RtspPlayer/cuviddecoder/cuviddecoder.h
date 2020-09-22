#ifndef CUVIDDECODER_H
#define CUVIDDECODER_H

#include <memory>
#include <vector>

#include "common.h"

//class Image{
//public:
//    int width = 0;
//    int height = 0;
//    Image(){}
//    Image(int w, int h){
//        set(w, h);
//    }
//    void set(int w, int h){
//        width = w;
//        height = h;
//        data.resize(w * h + w * h/2);
//    }

//    std::vector<uint8_t> data;
//};

//typedef std::shared_ptr<Image> PImage;

class CuvidPrivate;

class CuvidDecoder
{
public:

    enum ET{
        eH264,
        eHEVC
    };

    CuvidDecoder(ET et = eH264);
    ~CuvidDecoder();

    size_t maximumQueueSize();
    void setMaximumQueueSize(size_t size);

    bool isUpdateImage() const;
    void resetUpdateImage();

    bool decode(uint8_t *data, size_t size, PImage& image);

private:
    std::shared_ptr<CuvidPrivate> mD;
};

#endif // CUVIDDECODER_H
