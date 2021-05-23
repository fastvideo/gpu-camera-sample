#ifndef CAMERASTATISTICS_H
#define CAMERASTATISTICS_H

#include <QWidget>
#include <QTimer>
#include "GPUCameraBase.h"

namespace Ui {
class CameraStatistics;
}

class CameraStatistics : public QWidget
{
    Q_OBJECT

public:
    explicit CameraStatistics(QWidget *parent = nullptr);
    ~CameraStatistics();
    void setCamera(GPUCameraBase* cameraPtr){mCamera = cameraPtr;}
private:
    Ui::CameraStatistics *ui;
    QTimer mUpdateTimer;
    void UpdateStatPanel();
    GPUCameraBase *mCamera{nullptr};
};

#endif // CAMERASTATISTICS_H
