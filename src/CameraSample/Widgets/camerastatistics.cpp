#include "camerastatistics.h"
#include "ui_camerastatistics.h"

using CameraStatEnum = GPUCameraBase::cmrCameraStatistic  ;

CameraStatistics::CameraStatistics(QWidget *parent) :
   QWidget(parent),
   ui(new Ui::CameraStatistics),
   mUpdateTimer(this)
{
    ui->setupUi(this);

    // Set update timer
    connect(&mUpdateTimer, &QTimer::timeout, this, &CameraStatistics::UpdateStatPanel);
    mUpdateTimer.setSingleShot(false);
    // update every 1 second
    mUpdateTimer.setInterval(1000);
    mUpdateTimer.start();
}

CameraStatistics::~CameraStatistics()
{
    delete ui;
}

void CameraStatistics::UpdateStatPanel()
{
    // check if camera connected
    if(mCamera==nullptr)
        return;

    // Update the statistics
    uint64_t val=0;
    if(mCamera->GetStatistics(CameraStatEnum::statCurrFps100, val))
        ui->leCurrFPS->setText(QString("%1").arg(float(val)/100.));
    if(mCamera->GetStatistics(CameraStatEnum::statCurrFrameID, val))
        ui->leCurrFrameID->setText(QString("%1").arg(val));
    if(mCamera->GetStatistics(CameraStatEnum::statCurrTimestamp, val))
        ui->leCurrTimestamp->setText(QString("%1").arg(val));
    if(mCamera->GetStatistics(CameraStatEnum::statCurrTroughputMbs100, val))
        ui->leCurrThoughput->setText(QString("%1").arg(float(val)/100.));
    if(mCamera->GetStatistics(CameraStatEnum::statFramesDropped, val))
        ui->leFramesDropped->setText(QString("%1").arg(val));
    if(mCamera->GetStatistics(CameraStatEnum::statFramesIncomplete, val))
        ui->leFramesIncomplete->setText(QString("%1").arg(val));
    if(mCamera->GetStatistics(CameraStatEnum::statFramesTotal, val))
        ui->leFramesTotal->setText(QString("%1").arg(val));

}
