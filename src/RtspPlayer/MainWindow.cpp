#include "MainWindow.h"
#include "ui_mainwindow.h"

#include "DialogOpenServer.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));
    m_timer.start(300);

    //openServer("rtsp://127.0.0.1:1234/live.sdp");
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_actionOpen_RTSP_server_triggered()
{
    DialogOpenServer dlg;
    if(m_rtspServer.get())
        dlg.setUrl(m_rtspServer->url());
    if(dlg.exec()){
        openServer(dlg.url());
    }
}

void MainWindow::openServer(const QString &url)
{
    ui->widgetPlay->setReceiver(nullptr);
    m_rtspServer.reset(new RTSPServer);

    connect(m_rtspServer.get(), SIGNAL(startStopServer(bool)), this, SLOT(onStartStopServer(bool)), Qt::QueuedConnection);

    m_rtspServer->startServer(url, QMap<QString,QVariant>());
    ui->widgetPlay->setReceiver(m_rtspServer.get());
    ui->statusbar->showMessage("Try to open local server", 2000);
}

void MainWindow::openClient(const QString &url)
{
    ui->widgetPlay->setReceiver(nullptr);
    m_rtspServer.reset(new RTSPServer);

    connect(m_rtspServer.get(), SIGNAL(startStopServer(bool)), this, SLOT(onStartStopServer(bool)), Qt::QueuedConnection);

    QMap<QString,QVariant> params;
    params["client"] = 1;

    m_rtspServer->setUseCustomProtocol(ui->chb_use_custom_protocol->isChecked());
    m_rtspServer->startServer(url, params);
    ui->widgetPlay->setReceiver(m_rtspServer.get());
    ui->statusbar->showMessage("Try to open remote server", 2000);
}

void MainWindow::on_actionClose_RTSP_server_triggered()
{
    ui->widgetPlay->setReceiver(nullptr);
    m_rtspServer.reset();

    ui->statusbar->showMessage("Rtsp server is close");
}

void MainWindow::on_actionOpen_RTSP_client_triggered()
{
    DialogOpenServer dlg;
    dlg.setName("Open client");
    if(m_rtspServer.get())
        dlg.setUrl(m_rtspServer->url());
    else
        dlg.setUrl("");
    if(dlg.exec()){
        openClient(dlg.url());
    }
}

void MainWindow::on_pb_openRtsp_clicked()
{
    QString url = ui->le_rtsp_address->text();
    if(url.isEmpty()){
        ui->statusbar->showMessage("Url is empty");
        return;
    }
    if(ui->cb_rtspIsClient->isChecked()){
        openClient(url);
    }else{
        openServer(url);
    }
    m_rtspServer->setUseFastVideo(ui->chb_fastvideo->isChecked());
}

void MainWindow::onTimeout()
{
    if(m_rtspServer.get()){
        if(m_rtspServer->isError()){
            ui->statusbar->showMessage(m_rtspServer->errorStr());
        }else{
            if(m_rtspServer->done()){
                ui->statusbar->showMessage("Rtsp server is close");
            }
        }
        ui->lb_count_frames->setText(QString::number(m_rtspServer->framesCount()));
        ui->lb_fps->setText(QString::number(ui->widgetPlay->fps(), 'f', 1) + " frames/s");
        ui->lb_bitrate->setText(QString::number((ui->widgetPlay->bytesReaded() * 8)/1000, 'f', 1) + " kB/s");
    }
}

void MainWindow::on_chb_fastvideo_clicked(bool checked)
{
    if(m_rtspServer.get()){
        m_rtspServer->setUseFastVideo(checked);
    }
}

void MainWindow::on_pb_stopRtsp_clicked()
{
    on_actionClose_RTSP_server_triggered();
}

void MainWindow::onStartStopServer(bool start)
{
    if(start){
        ui->gtgWidget->start();
        ui->statusbar->showMessage("RTSP started");
    }else{
        ui->gtgWidget->stop();
        ui->statusbar->showMessage("RTSP stopped");
    }
}
