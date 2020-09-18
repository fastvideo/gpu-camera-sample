#include "MainWindow.h"
#include "ui_MainWindow.h"

#include "DialogOpenServer.h"

#include <QMapIterator>
#include <QSettings>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

	mRendererPtr.reset(new GLRenderer());

	mMediaViewer.reset(new GLImageViewer(mRendererPtr.data()));
	mContainerPtr.reset(QWidget::createWindowContainer(mMediaViewer.data()));
	mContainerPtr->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	ui->MediaViewerLayout->insertWidget(0, mContainerPtr.data());
	mContainerPtr->setMinimumSize(QSize(100, 100));
	mContainerPtr->setFocusPolicy(Qt::NoFocus);
	mRendererPtr->setRenderWnd(mMediaViewer.data());
	mRendererPtr->showImage(true);

    loadSettings();

    connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));
    m_timer.start(300);
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
        saveSettings();
        openServer(dlg.url());
    }
}

void MainWindow::openServer(const QString &url)
{
    m_rtspServer.reset(new RTSPServer(mRendererPtr.data()));

    connect(m_rtspServer.get(), SIGNAL(startStopServer(bool)), this, SLOT(onStartStopServer(bool)), Qt::QueuedConnection);

    m_rtspServer->startServer(url, QMap<QString,QVariant>());
    ui->statusbar->showMessage("Try to open local server", 2000);
}

void MainWindow::openClient(const QString &url)
{
    m_rtspServer.reset(new RTSPServer(mRendererPtr.data()));

    connect(m_rtspServer.get(), SIGNAL(startStopServer(bool)), this, SLOT(onStartStopServer(bool)), Qt::QueuedConnection);

    int h264id = 1;
    if(ui->rbCuvid->isChecked()){
        h264id = 1;
    }else if(ui->rbOtherAvailable->isChecked()){
        h264id = 2;
    }else if(ui->rbCuvidNV->isChecked()){
        h264id = 3;
    }

    QMap<QString,QVariant> params;
    params["client"] = 1;
	params["mjpeg_fastvideo"] = ui->rbFastvideoJpeg->isChecked();
    params["h264"] = h264id;
	params["ctp"] = ui->rbCtp->isChecked();

    m_rtspServer->startServer(url, params);
    ui->statusbar->showMessage("Try to open remote server", 2000);
}

void MainWindow::loadSettings()
{
    QSettings settings(QSettings::IniFormat, QSettings::UserScope, "rtspplayer");

    settings.beginGroup("pref");
    QString url = settings.value("url").toString();
    if(!url.isEmpty()){
        ui->le_rtsp_address->setText(url);
    }
    settings.endGroup();
}

void MainWindow::saveSettings()
{
    QSettings settings(QSettings::IniFormat, QSettings::UserScope, "rtspplayer");
    settings.beginGroup("pref");
    settings.setValue("url", ui->le_rtsp_address->text());
    settings.endGroup();
    settings.sync();
}

void MainWindow::on_actionClose_RTSP_server_triggered()
{
    m_rtspServer.reset();

	ui->gbDecodersH264->setEnabled(true);
	ui->gbTransportProtocol->setEnabled(true);

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
        saveSettings();
        openClient(dlg.url());
		ui->actionPlay->setChecked(true);
		ui->gtgWidget->start();
    }
}

void MainWindow::on_pb_openRtsp_clicked()
{
    QString url = ui->le_rtsp_address->text();
    if(url.isEmpty()){
        ui->statusbar->showMessage("Url is empty");
        return;
    }

    saveSettings();
    openClient(url);
}

void MainWindow::onTimeout()
{
    if(m_rtspServer.get() && m_rtspServer->isLive()){
        if(m_rtspServer->isError()){
            ui->statusbar->showMessage(m_rtspServer->errorStr());
        }else{
            if(m_rtspServer->done()){
                ui->statusbar->showMessage("Rtsp server is close");
            }
        }

		if(m_rtspServer->isMJpeg()){
			ui->gbDecodersH264->setVisible(false);
			ui->gbMJpegParameters->setVisible(true);
		}else{
			ui->gbDecodersH264->setVisible(true);
			ui->gbMJpegParameters->setVisible(false);
		}

		if(m_rtspServer->isServerOpened()){
			ui->gbDecodersH264->setEnabled(false);
			ui->gbTransportProtocol->setEnabled(false);
		}else{
			ui->gbDecodersH264->setEnabled(true);
			ui->gbTransportProtocol->setEnabled(true);
		}

        ui->lb_count_frames->setText(QString::number(m_rtspServer->framesCount()));
		ui->lb_fps->setText(QString::number(mRendererPtr->fps(), 'f', 1) + " frames/s");
		ui->lb_bitrate->setText(QString::number((mRendererPtr->bytesReaded() * 8)/1000, 'f', 1) + " kB/s");

		QString sdur;

		QMap<QString, double> durations = m_rtspServer->durations();

        if(mRendererPtr.data()){
            sdur += QString("Output image: %1x%2 pixels\n").arg(mRendererPtr->imageSize().width()).arg(mRendererPtr->imageSize().height());
        }

		{
			sdur += "Decoding: \n";

			QMapIterator<QString, double> it(durations);
			while(it.hasNext()){
				it.next();

				sdur += it.key() + " = " + QString::number(it.value(), 'f', 3) + " ms\n";
			}
		}

		{
//			sdur += "\nShow: \n";

//			durations = ui->widgetPlay->durations();
//			QMapIterator<QString, double> it(durations);
//			while(it.hasNext()){
//				it.next();

//				sdur += it.key() + " = " + QString::number(it.value(), 'f', 3) + " ms\n";
//			}
		}

		ui->lb_durations->setText(sdur);
	}else{
        if(m_rtspServer.get()){
            m_rtspServer.reset();
        }

        ui->actionPlay->setChecked(true);
        on_actionPlay_toggled(true);
		ui->gbDecodersH264->setEnabled(true);
		ui->gbTransportProtocol->setEnabled(true);
		ui->gbDecodersH264->setVisible(true);
		ui->gbMJpegParameters->setVisible(true);
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

void MainWindow::on_actionPlay_toggled(bool arg1)
{
	if(arg1)
	{
		if(m_rtspServer.get()){
			m_rtspServer->startDecode();
            ui->gtgWidget->start();
        }else{
            ui->gtgWidget->stop();
        }
		//ui->widgetPlay->start();
	}
	else
	{
		if(m_rtspServer.get()){
			m_rtspServer->stopDecode();
		}
//		ui->widgetPlay->stop();
		ui->gtgWidget->stop();
	}
}

void MainWindow::on_rbJpegTurbo_clicked(bool checked)
{
	if(checked){
		if(m_rtspServer.get()){
			m_rtspServer->setUseFastVideo(false);
		}
	}
}

void MainWindow::on_rbFastvideoJpeg_clicked(bool checked)
{
	if(checked){
		if(m_rtspServer.get()){
			m_rtspServer->setUseFastVideo(true);
		}
	}
}

void MainWindow::closeEvent(QCloseEvent *event)
{
	if(m_rtspServer){
		m_rtspServer.reset();
	}

	QMainWindow::closeEvent(event);
}
