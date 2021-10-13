#include "MIPICamera.h"

MIPICamera::MIPICamera()
{
    mCameraThread.setObjectName(QStringLiteral("MIPICameraThread"));
    moveToThread(&mCameraThread);
    mCameraThread.start();
}

MIPICamera::~MIPICamera()
{
    mCameraThread.quit();
    mCameraThread.wait(3000);
}

void MIPICamera::startStreaming()
{

}

bool MIPICamera::open(uint32_t devID)
{
    return false;
}

bool MIPICamera::start()
{
    mState = cstStreaming;
    emit stateChanged(cstStreaming);
    QTimer::singleShot(0, this, [this](){startStreaming();});
    return true;
}

bool MIPICamera::stop()
{
    mState = cstStopped;
    emit stateChanged(cstStopped);

    return true;
}

void MIPICamera::close()
{
    stop();
    //xiCloseDevice(hDevice);
    mState = cstClosed;
    emit stateChanged(cstClosed);
}

bool MIPICamera::getParameter(GPUCameraBase::cmrCameraParameter param, float &val)
{
    return false;
}

bool MIPICamera::setParameter(GPUCameraBase::cmrCameraParameter param, float val)
{
    return false;
}

bool MIPICamera::getParameterInfo(GPUCameraBase::cmrParameterInfo &info)
{
    return false;
}
