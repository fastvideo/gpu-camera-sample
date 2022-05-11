#ifndef MIPICAMERA_H
#define MIPICAMERA_H

#include "GPUCameraBase.h"

#include <QSharedPointer>

class CameraV4l2;

class MIPICamera : public GPUCameraBase
{
public:
    MIPICamera();
    ~MIPICamera();

    void startStreaming();

private:
    QSharedPointer<CameraV4l2> mCamera;

    // GPUCameraBase interface
public:
    bool open(uint32_t devID);
    bool start();
    bool stop();
    void close();
    bool getParameter(cmrCameraParameter param, float &val);
    bool setParameter(cmrCameraParameter param, float val);
    bool getParameterInfo(cmrParameterInfo &info);
};

#endif // MIPICAMERA_H
