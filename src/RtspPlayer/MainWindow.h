#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>

#include <QScopedPointer>
#include <QElapsedTimer>
#include "GLImageViewer.h"

#include "RTSPServer.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_actionOpen_RTSP_server_triggered();

    void on_actionClose_RTSP_server_triggered();

    void on_actionOpen_RTSP_client_triggered();

    void on_pb_openRtsp_clicked();

    void onTimeout();

    void on_pb_stopRtsp_clicked();

    void onStartStopServer(bool start);

	void on_actionPlay_toggled(bool arg1);

	void on_rbJpegTurbo_clicked(bool checked);

	void on_rbFastvideoJpeg_clicked(bool checked);

protected:
	void closeEvent(QCloseEvent* event) override;

private:
    Ui::MainWindow *ui;
    QTimer m_timer;

    std::unique_ptr<RTSPServer> m_rtspServer;

	QScopedPointer<QWidget> mContainerPtr;
	QScopedPointer<GLImageViewer> mMediaViewer;
	QSharedPointer<GLRenderer> mRendererPtr;

    void openServer(const QString &url);
    void openClient(const QString &url);

    void loadSettings();
    void saveSettings();
};

#endif // MAINWINDOW_H
