#ifndef GTGWIDGET_H
#define GTGWIDGET_H

#include <QWidget>

class GtGWidget : public QWidget
{
    Q_OBJECT
public:
    GtGWidget(QWidget* parent = nullptr);
    void start(){setAnimating(true);}
    void stop(){setAnimating(false);}

private slots:

protected:
    void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;

private:
    void setAnimating(bool enabled);
    bool mAnimating = false;
    qint64 mLastTime = 0;
    qint64 mStartTime = 0;
    int mTimerInterval = 16;
};

#endif // GTGWIDGET_H
