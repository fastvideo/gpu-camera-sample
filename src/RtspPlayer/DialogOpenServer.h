#ifndef DIALOGOPENSERVER_H
#define DIALOGOPENSERVER_H

#include <QDialog>

namespace Ui {
class DialogOpenServer;
}

class DialogOpenServer : public QDialog
{
    Q_OBJECT

public:
    explicit DialogOpenServer(QWidget *parent = nullptr);
    ~DialogOpenServer();

    void setName(const QString& name);

    void setUrl(const QString url);
    QString url();

private slots:
    void on_buttonBox_accepted();

private:
    Ui::DialogOpenServer *ui;

    QString m_url;
};

#endif // DIALOGOPENSERVER_H
