/*
 Copyright 2011-2019 Fastvideo, LLC.
 All rights reserved.

 This file is a part of the GPUCameraSample project
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 3. Any third-party SDKs from that project (XIMEA SDK, Fastvideo SDK, etc.) are licensed on different terms. Please see their corresponding license terms.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 The views and conclusions contained in the software and documentation are those
 of the authors and should not be interpreted as representing official policies,
 either expressed or implied, of the FreeBSD Project.
*/

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

void CameraSetupWidget::setCamera(GPUCameraBase* cameraPtr)
{
    mCameraPtr = cameraPtr;

    if(mCameraPtr == nullptr)
    {
        ui->spnFrameRate->setEnabled(false);
        ui->spnExposureTime->setEnabled(false);
    }
    else if(mCameraPtr->state() == GPUCameraBase::cstClosed)
    {
        ui->spnFrameRate->setEnabled(false);
        ui->spnExposureTime->setEnabled(false);
    }
    else
    {
        ui->spnFrameRate->setEnabled(true);
        ui->spnExposureTime->setEnabled(true);

        connect(mCameraPtr, SIGNAL(stateChanged(GPUCameraBase::cmrCameraState)),
                this, SLOT(onCameraStateChanged(GPUCameraBase::cmrCameraState)));

        GPUCameraBase::cmrParameterInfo info(GPUCameraBase::prmExposureTime);
        float val = 0;

        mCameraPtr->getParameterInfo(info);
        QSignalBlocker b(ui->spnExposureTime);
        ui->spnExposureTime->setMaximum(info.max);
        ui->spnExposureTime->setMinimum(info.min);
        ui->spnExposureTime->setSingleStep(info.increment);
        if(mCameraPtr->getParameter(GPUCameraBase::prmExposureTime, val))
        {
            ui->spnExposureTime->setValue((int)val);
        }

        QSignalBlocker b1(ui->spnFrameRate);
        info.param = GPUCameraBase::prmFrameRate;
        mCameraPtr->getParameterInfo(info);
        ui->spnFrameRate->setMaximum(info.max);
//        ui->spnFrameRate->setMinimum(info.min);
        ui->spnFrameRate->setSingleStep(info.increment);
        if(mCameraPtr->getParameter(GPUCameraBase::prmFrameRate, val))
        {
            ui->spnFrameRate->setValue((double)val);
        }
    }
}

void CameraSetupWidget::setExposureCamera(float value)
{
    ui->spnExposureTime->setValue(value);
}
void CameraSetupWidget::on_spnFrameRate_valueChanged(double arg1)
{
    if(mCameraPtr == nullptr)
        return;
    if(mCameraPtr->state() == GPUCameraBase::cstClosed)
        return;

    mCameraPtr->setParameter(GPUCameraBase::prmFrameRate, arg1);


    QSignalBlocker b(ui->spnExposureTime);
    float val = 0;
    if(mCameraPtr->getParameter(GPUCameraBase::prmExposureTime, val))
    {
        ui->spnExposureTime->setValue((int)val);
    }
}

void CameraSetupWidget::on_spnExposureTime_valueChanged(int arg1)
{
    if(mCameraPtr == nullptr)
        return;

    if(mCameraPtr->state() == GPUCameraBase::cstClosed)
        return;

    mCameraPtr->setParameter(GPUCameraBase::prmExposureTime, arg1);

    QSignalBlocker b1(ui->spnFrameRate);
    float val = 0;
    if(mCameraPtr->getParameter(GPUCameraBase::prmFrameRate, val))
    {
        ui->spnFrameRate->setValue((double)val);
    }
}

void CameraSetupWidget::onCameraStateChanged(GPUCameraBase::cmrCameraState newState)
{
    if(newState == GPUCameraBase::cstClosed)
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
