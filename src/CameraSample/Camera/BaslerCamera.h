#ifndef BASLERCAMERA_H
#define BASLERCAMERA_H

#ifdef SUPPORT_BASLER

#include <QObject>
#include "GPUCameraBase.h"
#include <pylon/PylonIncludes.h>

class BaslerCamera : public GPUCameraBase
{
public:
    BaslerCamera();

    ~BaslerCamera();

    virtual bool open(uint32_t devID);
    virtual bool start();
    virtual bool stop();
    virtual void close();

    virtual bool getParameter(cmrCameraParameter param, float& val);
    virtual bool setParameter(cmrCameraParameter param, float val);
    virtual bool getParameterInfo(cmrParameterInfo& info);
private:
    void startStreaming();
    void UpdateStatistics(const Pylon::CGrabResultPtr pBuff);
    QScopedPointer<Pylon::CInstantCamera> mCamera;
//      Pylon::CInstantCamera mCamera;

};
#endif SUPPORT_BASLER
#endif // BASLERCAMERA_H
