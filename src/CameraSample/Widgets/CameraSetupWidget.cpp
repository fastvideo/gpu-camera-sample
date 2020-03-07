#include "CameraSetupWidget.h"
#include "ui_CameraSetupWidget.h"

CameraSetupWidget::CameraSetupWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::CameraSetupWidget)
{
    ui->setupUi(this);
}

CameraSetupWidget::~CameraSetupWidget()
{
    delete ui;
}

void CameraSetupWidget::setCamera(CameraBase* cameraPtr)
{
    mCameraPtr = cameraPtr;

    if(mCameraPtr == nullptr)
    {
        ui->spnFrameRate->setEnabled(false);
        ui->spnExposureTime->setEnabled(false);
    }
    else if(mCameraPtr->state() == CameraBase::cstClosed)
    {
        ui->spnFrameRate->setEnabled(false);
        ui->spnExposureTime->setEnabled(false);
    }
    else
    {
        ui->spnFrameRate->setEnabled(true);
        ui->spnExposureTime->setEnabled(true);

        connect(mCameraPtr, SIGNAL(stateChanged(CameraBase::cmrCameraState)),
                this, SLOT(onCameraStateChanged(CameraBase::cmrCameraState)));

        CameraBase::cmrParameterInfo info(CameraBase::prmExposureTime);
        float val = 0;

        mCameraPtr->getParameterInfo(info);
        QSignalBlocker b(ui->spnExposureTime);
        ui->spnExposureTime->setMaximum(info.max);
        ui->spnExposureTime->setMinimum(info.min);
        ui->spnExposureTime->setSingleStep(info.increment);
        if(mCameraPtr->getParameter(CameraBase::prmExposureTime, val))
        {
            ui->spnExposureTime->setValue((int)val);
        }

        QSignalBlocker b1(ui->spnFrameRate);
        info.param = CameraBase::prmFrameRate;
        mCameraPtr->getParameterInfo(info);
        ui->spnFrameRate->setMaximum(info.max);
//        ui->spnFrameRate->setMinimum(info.min);
        ui->spnFrameRate->setSingleStep(info.increment);
        if(mCameraPtr->getParameter(CameraBase::prmFrameRate, val))
        {
            ui->spnFrameRate->setValue((double)val);
        }
    }
}
void CameraSetupWidget::on_spnFrameRate_valueChanged(double arg1)
{
    if(mCameraPtr == nullptr)
        return;
    if(mCameraPtr->state() == CameraBase::cstClosed)
        return;

    mCameraPtr->setParameter(CameraBase::prmFrameRate, arg1);


    QSignalBlocker b(ui->spnExposureTime);
    float val = 0;
    if(mCameraPtr->getParameter(CameraBase::prmExposureTime, val))
    {
        ui->spnExposureTime->setValue((int)val);
    }
}

void CameraSetupWidget::on_spnExposureTime_valueChanged(int arg1)
{
    if(mCameraPtr == nullptr)
        return;

    if(mCameraPtr->state() == CameraBase::cstClosed)
        return;

    mCameraPtr->setParameter(CameraBase::prmExposureTime, arg1);

    QSignalBlocker b1(ui->spnFrameRate);
    float val = 0;
    if(mCameraPtr->getParameter(CameraBase::prmFrameRate, val))
    {
        ui->spnFrameRate->setValue((double)val);
    }
}

void CameraSetupWidget::onCameraStateChanged(CameraBase::cmrCameraState newState)
{
    if(newState == CameraBase::cstClosed)
    {
        ui->spnFrameRate->setEnabled(false);
        ui->spnExposureTime->setEnabled(false);
    }
    else
    {
        ui->spnFrameRate->setEnabled(true);
        ui->spnExposureTime->setEnabled(true);
    }
//    else if(newState == CameraBase::cstStopped)
//    {

//    }
//    else if(newState == CameraBase::cstStreaming)
//    {

//    }
}
