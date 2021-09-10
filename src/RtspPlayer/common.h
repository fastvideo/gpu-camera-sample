#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <memory>

#include <chrono>

#include <QMap>
#include <QMapIterator>

typedef std::vector< unsigned char > bytearray;

class RTSPImage: public std::enable_shared_from_this<RTSPImage>{
public:
    enum TYPE{YUV, NV12, P010, RGB, GRAY, CUDA_RGB, CUDA_GRAY};

    RTSPImage();
    RTSPImage(int w, int h, TYPE tp);
    RTSPImage(const RTSPImage& o);
    ~RTSPImage();

	void setYUV(uint8_t *data[], int linesize[], int w, int h);
	void setNV12(uint8_t *data[], int linesize[], int w, int h);
    void setP010(uint8_t *data[], int linesize[], int w, int h);
    bool setCudaRgb(int w, int h);
	bool setCudaGray(int w, int h);
	void setYUV(int w, int h);
	void setNV12(int w, int h);
    void setP010(int w, int h);
    void setRGB(int w, int h);
	void setGray(int w, int h);

	void releaseCudaRgbBuffer();

	bool empty() const;

	bytearray yuv;
    bytearray rgb;
    TYPE type = YUV;
    int width = 0;
    int height = 0;
	void *cudaRgb = nullptr;
	size_t cudaSize = 0;
private:

};

typedef std::shared_ptr<RTSPImage> PImage;

class AbstractReceiver{
public:
    virtual ~AbstractReceiver(){}

    virtual uint64_t bytesReaded() = 0;
};


#ifdef _MSC_VER
typedef std::chrono::steady_clock::time_point timepoint;
#else
typedef std::chrono::system_clock::time_point timepoint;
#endif

inline timepoint getNow()
{
	return std::chrono::high_resolution_clock::now();
}

inline double getDuration(timepoint start)
{
	auto dur = std::chrono::high_resolution_clock::now() - start;
	double duration = std::chrono::duration_cast<std::chrono::microseconds>(dur).count()/1000.;
	return duration;
}

template<typename T>
inline QMap<QString, T> mergeMaps(const QMap<QString, T>& first, const QMap<QString, T>& second)
{
	QMap<QString, T> res = first;
	QMapIterator<QString, T> it(second);

	while(it.hasNext()){
		it.next();
		res[it.key()] = it.value();
	}
	return res;
}

#endif // COMMON_H
