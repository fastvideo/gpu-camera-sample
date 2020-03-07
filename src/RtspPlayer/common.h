#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <memory>

typedef std::vector< unsigned char > bytearray;

class Image{
public:
    enum TYPE{YUV, RGB, GRAY};

    Image(){

    }
    Image(int w, int h, TYPE tp){
        if(tp == YUV)
            setYUV(w, h);
        else if(tp == RGB)
            setRGB(w, h);
        else if(tp == GRAY)
            setGray(w, h);
    }

    void setYUV(uint8_t *data[], int linesize[], int w, int h){
        type = YUV;
        width = w;
        height = h;
        size_t size1 = static_cast<size_t>(linesize[0] * h);
        size_t size2 = static_cast<size_t>(linesize[1] * h/2);
        size_t size3 = static_cast<size_t>(linesize[2] * h/2);
        Y.resize(size1);
        U.resize(size2);
        V.resize(size3);

        std::copy(data[0], data[0] + size1, Y.data());
        std::copy(data[1], data[1] + size2, U.data());
        std::copy(data[2], data[2] + size3, V.data());
    }

    void setNV12(uint8_t *data[], int linesize[], int w, int h){
        type = YUV;
        width = w;
        height = h;
        size_t size1 = static_cast<size_t>(linesize[0] * h);
        size_t size2 = static_cast<size_t>(linesize[1]/2 * h/2);
        size_t size3 = static_cast<size_t>(linesize[1]/2 * h/2);
        Y.resize(size1);
        U.resize(size2);
        V.resize(size3);

        std::copy(data[0], data[0] + size1, Y.data());

        for(size_t x = 0; x < size2; ++x){
            U[x] = data[1][x * 2 + 0];
            V[x] = data[1][x * 2 + 1];
        }
    }

    void setYUV(int w, int h){
        type = YUV;
        width = w;
        height = h;
        Y.resize(w * h);
        U.resize(w/2 * h/2);
        V.resize(w/2 * h/2);
    }
    void setRGB(int w, int h){
        type = RGB;
        width = w;
        height = h;
        rgb.resize(w * h * 3);
    }
    void setGray(int w, int h){
        type = GRAY;
        width = w;
        height = h;
        rgb.resize(w * h);
    }

    bool empty() const{
        return width == 0 || height == 0;
    }

    bytearray Y;
    bytearray U;
    bytearray V;
    bytearray rgb;
    TYPE type = YUV;
    int width = 0;
    int height = 0;
private:

};

typedef std::shared_ptr<Image> PImage;

class AbstractReceiver{
public:
    virtual ~AbstractReceiver(){}

    virtual bool isFrameExists() const = 0;
    virtual PImage takeFrame() = 0;
    virtual uint64_t bytesReaded() = 0;
};

#endif // COMMON_H
