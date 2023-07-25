/*
 Copyright 2011-2019 Fastvideo, LLC.
 All rights reserved.

 This file is a part of the GPUCameraSample project
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 3. Any third-party SDKs from that project (XIMEA SDK, Fastvideo SDK, etc.) are licensed on different terms. Please see their corresponding license terms.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 The views and conclusions contained in the software and documentation are those
 of the authors and should not be interpreted as representing official policies,
 either expressed or implied, of the FreeBSD Project.
*/

#include "GtGWidget.h"

#include <QApplication>
#include <QList>
#include <QScreen>
#include <QScreen>
#include <QDateTime>
#include <QImage>
#include <QPainter>
#include <QTimer>
#include <QPainterPath>

#if QT_VERSION_MAJOR < 6
#include <QDesktopWidget>
#endif

GtGWidget::GtGWidget(QWidget* parent) :
    QWidget(parent)
{
    mLastTime = QDateTime::currentMSecsSinceEpoch();
    mStartTime = mLastTime;

#if QT_VERSION_MAJOR >= 6
    QScreen *screen = window()->screen();
#else
    QScreen* screen = QGuiApplication::screens().at( QApplication::desktop()->screenNumber());
#endif

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
    p.setBrush(QBrush(Qt::white));
    p.setPen(QPen(Qt::white, 1));
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
