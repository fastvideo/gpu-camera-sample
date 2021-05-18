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

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>

#include "CUDAProcessorBase.h"
#include "FrameBuffer.h"
#include "GPUCameraBase.h"
#include "GLImageViewer.h"

class GLImageViewer;
class RawProcessor;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    typedef enum
    {
        ogLinear = 0,
        ogsRGB
    } OutputGamma;

    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

protected:
    void customEvent(QEvent* event) Q_DECL_OVERRIDE;
    void closeEvent(QCloseEvent* event) Q_DECL_OVERRIDE;
private slots:
    //Zoom
    void onZoomChanged(qreal zoom);
    void on_chkZoomFit_toggled(bool checked);
    void on_sldZoom_valueChanged(int value);
    void on_btnResetZoom_clicked();

    //Debayer
    void on_cboBayerPattern_currentIndexChanged(int index);
    void on_cboBayerType_currentIndexChanged(int index);

    //Exposure correction
    void on_sldEV_valueChanged(int value);
    void on_btnResetEV_clicked();

    //Denoise
    void onDenoiseStateChanged(bool on);
    void onThresholdTypeChanged(int newType);
    void onWaveletTypeChanged(int newType);
    void onYThresholdChanged(float newThreshold);
    void onCbThresholdChanged(float newThreshold);
    void onShrinkageChanged(int newSrinkage);
    void onDenoiseParamsChanged();

    //White balance
    void on_sldRed_valueChanged(int value);
    void on_sldGreen_valueChanged(int value);
    void on_sldBlue_valueChanged(int value);
    void on_btnResetRed_clicked();
    void on_btnResetGreen_clicked();
    void on_btnResetBlue_clicked();

    //Gamma
    void on_cboGamma_currentIndexChanged(int index);

    //Toolbar
    void on_actionOpenCamera_triggered();
    void on_actionRecord_toggled(bool arg1);
    void on_actionExit_triggered();
    void on_actionWB_picker_toggled(bool arg1);
    void on_actionPlay_toggled(bool arg1);

    //CUDA processor
    QString getErrDescription(fastStatus_t code);
    void    onGPUError();
    void    onGPUFinished();

    //BPC
    void on_chkBPC_toggled(bool checked);

    //WB
    void onNewWBFromPoint(const QPoint& pt);

    //Camera
    void openCamera(uint32_t devID);
    void openPGMFile(bool isBayer = true);
    void initNewCamera(GPUCameraBase* cmr, uint32_t devID);
    void onCameraStateChanged(GPUCameraBase::cmrCameraState newState);
    void on_actionOpenBayerPGM_triggered();
    void on_actionOpenGrayPGM_triggered();


    void on_btnGetOutPath_clicked();
    void on_btnGetFPNFile_clicked();
    void on_btnGetGrayFile_clicked();
    void on_chkSAM_toggled(bool checked);

    //RTSP
    void on_btnStartRtspServer_clicked();
    void on_btnStopRtspServer_clicked();
    void onTimeoutStatusRtsp();


	void on_cboFormatEnc_currentIndexChanged(int index);
    void on_actionShowImage_triggered(bool checked);


private:
    Ui::MainWindow *ui;

    QLabel* mStatusLabel;
    QLabel* mFpsLabel;

    QScopedPointer<RawProcessor> mProcessorPtr;
    QScopedPointer<GPUCameraBase> mCameraPtr;

    QScopedPointer<QWidget> mContainerPtr;
    QScopedPointer<GLImageViewer> mMediaViewer;
    QSharedPointer<GLRenderer> mRendererPtr;

    CUDAProcessorOptions mOptions;
    QVector<unsigned short> mGammaCurve;
    QTimer mTimerStatusRtsp;

    void delayInit();
    void raw2Rgb(bool update = true, bool init = false);
    void updateAll();
    void updateOptions(CUDAProcessorOptions& opts);

    void readSettings();
    void writeSettings();
};

#endif // MAINWINDOW_H
