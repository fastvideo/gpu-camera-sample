#ifndef GENICAMCAMERA_H
#define GENICAMCAMERA_H

#ifdef SUPPORT_GENICAM

#include "GPUCameraBase.h"
#include <rc_genicam_api/system.h>
#include <rc_genicam_api/interface.h>
#include <rc_genicam_api/device.h>
#include <rc_genicam_api/stream.h>
#include <rc_genicam_api/config.h>
#include <rc_genicam_api/pixel_formats.h>
#include <QObject>

class GeniCamCamera : public GPUCameraBase
{
public:
    GeniCamCamera();
    ~GeniCamCamera();

    virtual bool open(uint32_t devID);
    virtual bool start();
    virtual bool stop();
    virtual void close();

    virtual bool getParameter(cmrCameraParameter param, float& val);
    virtual bool setParameter(cmrCameraParameter param, float val);
    virtual bool getParameterInfo(cmrParameterInfo& info);
protected:

private:
    bool mStreaming = false;
    void startStreaming();
    std::shared_ptr<rcg::Device> mDevice;

    void UpdateStatistics(const rcg::Buffer* pb);


};

#endif // SUPPORT_GENICAM

#endif // GENICAMCAMERA_H
