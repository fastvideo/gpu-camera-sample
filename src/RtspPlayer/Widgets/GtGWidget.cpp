#include "GtGWidget.h"

#include <QApplication>
#include <QList>
#include <QScreen>
#include <QDesktopWidget>
#include <QDateTime>
#include <QImage>
#include <QPainter>
#include <QTimer>
#include <QPainterPath>

GtGWidget::GtGWidget(QWidget* parent) :
    QWidget(parent)
{
    mLastTime = QDateTime::currentMSecsSinceEpoch();
    mStartTime = mLastTime;

    QScreen* screen = QGuiApplication::screens().at(QApplication::desktop()->screenNumber());
    if(screen)
    {
        mTimerInterval = 1000. / screen->refreshRate();
    }
}

void GtGWidget::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event)

    QPainter p;
    p.begin(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.setBrush(QBrush(Qt::black));
    p.setPen(QPen(Qt::black, 1));
    QPainterPath textPath;
    qint64 lastTime = mAnimating ? QDateTime::currentMSecsSinceEpoch() : mLastTime;

    QString str;
    if(lastTime > mStartTime)
        str = QString::number(lastTime - mStartTime);

    textPath.addText(0, 0, QFont("Arial"), str);
    const QRectF rcText = textPath.boundingRect();
    p.setWindow(rcText.left() - 5, rcText.top() - 5, rcText.width() + 10, rcText.height() + 10);
    p.drawPath(textPath);
    p.end();
    if(mAnimating)
    {
        mLastTime = lastTime;
        QTimer::singleShot(mTimerInterval, this, [this](){update();});
    }
}

void GtGWidget::setAnimating(bool enabled)
{
    mAnimating = enabled;
    if(mAnimating)
    {
        mStartTime = QDateTime::currentMSecsSinceEpoch();
        update();
    }
}
