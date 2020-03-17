#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>

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

    void on_chb_fastvideo_clicked(bool checked);

    void on_pb_stopRtsp_clicked();

    void onStartStopServer(bool start);

	void on_actionPlay_toggled(bool arg1);

private:
    Ui::MainWindow *ui;
    QTimer m_timer;

    std::unique_ptr<RTSPServer> m_rtspServer;

    void openServer(const QString &url);
    void openClient(const QString &url);
};

#endif // MAINWINDOW_H
