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

#include "DenoiseController.h"
#include "ui_DenoiseController.h"
#include <iterator>
#include <cmath>

DenoiseController::DenoiseController(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::DenoiseController)
{
    ui->setupUi(this);
    mBpp = 8;

    ui->cboWaveletType->addItem(QStringLiteral("CDF97"), FAST_WAVELET_CDF97);
    ui->cboWaveletType->addItem(QStringLiteral("CDF53"), FAST_WAVELET_CDF53);

    connect(ui->chkUseDenoise, SIGNAL(clicked(bool)), this, SIGNAL(stateChanged(bool)));

    mSliders << ui->sldY0;
    mSliders << ui->sldCb0;
    mSliders << ui->sldCr0;

    mSliders << ui->sldY1;
    mSliders << ui->sldCb1;
    mSliders << ui->sldCr1;

    mSliders << ui->sldY2;
    mSliders << ui->sldCb2;
    mSliders << ui->sldCr2;

    mSliders << ui->sldY3;
    mSliders << ui->sldCb3;
    mSliders << ui->sldCr3;

    mSliders << ui->sldY4;
    mSliders << ui->sldCb4;
    mSliders << ui->sldCr4;


    for(int i = 0; i < 15; i++)
        connect(mSliders[i], SIGNAL(valueChanged(int)), this, SLOT(onSliderValueChanged(int)));

    ui->tabWidget->setTabEnabled(4, !ui->chkSyncColor->isChecked());
}

DenoiseController::~DenoiseController()
{
    delete ui;
}


void DenoiseController::getDenoiseParams(fastDenoiseParameters_t &params, bool& denoiseState)
{
    params.dwt_levels = ui->spnDecompLevel->value();
    params.threshold[0] = float(ui->sldYThreshold->value()) / 100;
    params.enhance[0] = 1.;//(float)ui->sldYBlur->value() / 100;

    std::fill(std::begin(params.threshold_per_level), std::end(params.threshold_per_level),1.f);

    if(ui->chkSyncColor->isChecked())
    {
        params.threshold[1] = params.threshold[2] = float(ui->sldCbThreshold->value()) / 100;
        params.enhance[1] = 1.;//params.enhance[2] = (float)ui->sldCbBlur->value() / 100;

        for(int i = 0; i < 15; i+=3)
        {
            double val = mSliders[i]->value();
            params.threshold_per_level[i] = float(pow(10, val / 100));
        }
        for(int i = 1; i < 15; i+=3)
        {
            double val = mSliders[i]->value();
            params.threshold_per_level[i] = float(pow(10, val / 100));
            params.threshold_per_level[i + 1] = float(pow(10, val / 100));
        }
    }
    else
    {
        params.threshold[1] = float(ui->sldCbThreshold->value()) / 100;
        params.threshold[2] = float(ui->sldCrThreshold->value()) / 100;

        params.enhance[1] = 1.;
        params.enhance[2] = 1.;


        for(int i = 0; i < 15; i++)
        {
            double val = mSliders[i]->value();
            params.threshold_per_level[i] = float(pow(10, val / 100));
        }
    }

    denoiseState = ui->chkUseDenoise->isChecked();
}

void DenoiseController::getStaticDenoiseParams(fastDenoiseStaticParameters_t &params)
{
    Globals::validateStaticDenoiseParams(params);

    params.function = fastDenoiseThresholdFunctionType_t(ui->cboThresholdType->currentIndex() + 1);
    params.wavelet = fastWaveletType_t(ui->cboWaveletType->currentData().toInt());
}

void DenoiseController::setStaticDenoiseParams(fastDenoiseStaticParameters_t &params)
{
    Globals::validateStaticDenoiseParams(params);

    QSignalBlocker b1(ui->cboThresholdType);
    ui->cboThresholdType->setCurrentIndex(params.function - 1);

    QSignalBlocker b2(ui->cboWaveletType);
    int idx = ui->cboWaveletType->findData(params.wavelet);
    if(idx < 0) idx = 0;
    ui->cboWaveletType->setCurrentIndex(idx);
}

void DenoiseController::setDenoiseParams(const fastDenoiseParameters_t &params, bool denoiseState)
{
    QSignalBlocker b1(ui->spnDecompLevel);
    int val = params.dwt_levels;
    ui->spnDecompLevel->setValue(val);

    QSignalBlocker b2(ui->sldYThreshold);
    ui->sldYThreshold->setValue(int(params.threshold[0] * 100));
    ui->lblCurIntensity->setText(QString::number(double(params.threshold[0]), 'f', 2));

    QSignalBlocker b3(ui->sldCbThreshold);
    ui->sldCbThreshold->setValue(int(params.threshold[1] * 100));
    ui->lblCurCb->setText(QString::number(double(params.threshold[1]), 'f', 2));

    QSignalBlocker b4(ui->sldCrThreshold);
    ui->sldCrThreshold->setValue(int(params.threshold[2] * 100));
    ui->lblCurCr->setText(QString::number(double(params.threshold[2]), 'f', 2));

    for(int i = 0; i < 15; i++)
    {
        QSignalBlocker b(mSliders[i]);
        mSliders[i]->setValue(int(log10(params.threshold_per_level[i]) * 100));
    }

    QSignalBlocker b5(ui->chkUseDenoise);
    ui->chkUseDenoise->setChecked(denoiseState);
}

int DenoiseController::getBpp() const
{
    return mBpp;
}

void DenoiseController::setBpp(int value)
{
    if(value <= 8)
        mBpp = 8;
    else
        mBpp = 16;
}

void DenoiseController::on_cboThresholdType_currentIndexChanged(int index)
{
    emit thresholdTypeChanged(index + 1);
}


void DenoiseController::on_cboWaveletType_currentIndexChanged(int index)
{
    Q_UNUSED(index);
    emit waveletTypeChanged(ui->cboWaveletType->currentData().toInt());
}

void DenoiseController::on_sldCbThreshold_valueChanged(int value)
{
    ui->lblCurCb->setText(QString::number(double(value) / 100, 'f', 2));
    emit paramsChanged();
}

void DenoiseController::on_sldCrThreshold_valueChanged(int value)
{
    ui->lblCurCr->setText(QString::number(double(value) / 100, 'f', 2));
    emit paramsChanged();
}

void DenoiseController::on_sldYThreshold_valueChanged(int value)
{
    ui->lblCurIntensity->setText(QString::number(double(value) / 100, 'f', 2));
    emit paramsChanged();
}

void DenoiseController::on_spnMaxY_valueChanged(double arg1)
{
    ui->sldYThreshold->setMaximum(int(arg1 * 100));
}

void DenoiseController::on_spnMaxColor_valueChanged(double arg1)
{
    ui->sldCbThreshold->setMaximum(int(arg1 * 100));
    ui->sldCrThreshold->setMaximum(int(arg1 * 100));
}

void DenoiseController::onSliderValueChanged(int value)
{
    Q_UNUSED(value);
    emit paramsChanged();
}

void DenoiseController::on_btnResetY_clicked()
{
    for(int i = 0; i < 15; i+=3)
    {
        mSliders[i]->setValue(0);
    }
}

void DenoiseController::on_btnResetCb_clicked()
{
    for(int i = 1; i < 15; i+=3)
    {
        mSliders[i]->setValue(0);
    }
}

void DenoiseController::on_btnResetCr_clicked()
{
    for(int i = 2; i < 15; i+=3)
    {
        mSliders[i]->setValue(0);
    }
}

void DenoiseController::on_chkSyncColor_toggled(bool checked)
{
    ui->tabWidget->setTabEnabled(4, !checked);
    emit paramsChanged();
}
