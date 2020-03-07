#ifndef CAMERASETUPWIDGET_H
#define CAMERASETUPWIDGET_H

#include <QWidget>
#include "CameraBase.h"

namespace Ui {
class CameraSetupWidget;
}

class CameraSetupWidget : public QWidget
{
    Q_OBJECT

public:
    explicit CameraSetupWidget(QWidget *parent = 0);
    ~CameraSetupWidget();

    void setCamera(CameraBase* cameraPtr);

private slots:
    void on_spnFrameRate_valueChanged(double arg1);
    void on_spnExposureTime_valueChanged(int arg1);
    void onCameraStateChanged(CameraBase::cmrCameraState newState);
private:
    Ui::CameraSetupWidget *ui;
    CameraBase* mCameraPtr = nullptr;

};

#endif // CAMERASETUPWIDGET_H
