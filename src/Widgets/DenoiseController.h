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

#ifndef DENOISECONTROLLER_H
#define DENOISECONTROLLER_H

#include <QWidget>
#include <QSlider>
#include "fastvideo_denoise.h"
#include "Globals.h"

namespace Ui {
class DenoiseController;
}

class DenoiseController : public QWidget
{
    Q_OBJECT

public:
    explicit DenoiseController(QWidget *parent = nullptr);
    ~DenoiseController() override;
    void getDenoiseParams(denoise_parameters_t &params, bool& denoiseState);
    void setDenoiseParams(const denoise_parameters_t& params, bool denoiseState);
    void getStaticDenoiseParams(denoise_static_parameters_t &params);
    void setStaticDenoiseParams(denoise_static_parameters_t &params);
    int getBpp() const;
    void setBpp(int value);

signals:
    void stateChanged(bool on);
    void thresholdTypeChanged(int newType);
    void waveletTypeChanged(int newType);
    void shrinkageChanged(int newShrinkage);
    void yThresholdChanged(float newThreshold);
    void cbThresholdChanged(float newThreshold);
    void paramsChanged();

private slots:
    void on_cboThresholdType_currentIndexChanged(int index);
    void on_cboWaveletType_currentIndexChanged(int index);
    void on_sldCbThreshold_valueChanged(int value);
    void on_sldYThreshold_valueChanged(int value);
    void on_spnMaxY_valueChanged(double arg1);
    void on_spnMaxColor_valueChanged(double arg1);
    void onSliderValueChanged(int value);
    void on_sldCrThreshold_valueChanged(int value);
    void on_btnResetY_clicked();
    void on_btnResetCb_clicked();
    void on_btnResetCr_clicked();
    void on_chkSyncColor_toggled(bool checked);

private:
    Ui::DenoiseController *ui;
    int mBpp;

    QVector<QSlider*> mSliders;
};

#endif // DENOISECONTROLLER_H
