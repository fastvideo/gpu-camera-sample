#include "DialogOpenServer.h"
#include "ui_DialogOpenServer.h"

DialogOpenServer::DialogOpenServer(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DialogOpenServer)
{
    ui->setupUi(this);
}

DialogOpenServer::~DialogOpenServer()
{
    delete ui;
}

void DialogOpenServer::setName(const QString &name)
{
    setWindowTitle(name);
}

void DialogOpenServer::setUrl(const QString url)
{
    m_url = url;
    if(m_url.isEmpty()){
        m_url = "rtsp://127.0.0.1:1234/live.sdp";
    }
    ui->le_address->setText(m_url);
}

QString DialogOpenServer::url()
{
    return m_url;
}

void DialogOpenServer::on_buttonBox_accepted()
{
    m_url = ui->le_address->text();
}
