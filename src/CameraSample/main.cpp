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

#include "MainWindow.h"
#include <QApplication>
#include <QStyleFactory>
#include <QMessageBox>
#include "version.h"

int main(int argc, char *argv[])
{
#if QT_VERSION >= 0x050600
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

    QApplication a(argc, argv);

    QCoreApplication::setOrganizationName(QStringLiteral(APP_ORGANIZATION_NAME));
    QCoreApplication::setOrganizationDomain(QStringLiteral(APP_ORGANIZATION_DOMAIN));
    QCoreApplication::setApplicationName(QStringLiteral(MAIN_APPLICATION_NAME));
    QCoreApplication::setApplicationVersion(QStringLiteral(APP_VERSION_STRING));
    QCoreApplication::addLibraryPath(QStringLiteral("."));
    QCoreApplication::addLibraryPath(QCoreApplication::applicationDirPath());

    QApplication::setStyle(QStyleFactory::create(QStringLiteral("Fusion")));

    QPalette darkPalette;
    darkPalette.setColor(QPalette::Window, QColor(64,64,64));
    darkPalette.setColor(QPalette::WindowText, Qt::white);
    darkPalette.setColor(QPalette::Base, QColor(64,64,64));
    darkPalette.setColor(QPalette::AlternateBase, QColor(64,64,64));
    darkPalette.setColor(QPalette::ToolTipBase, Qt::white);
    darkPalette.setColor(QPalette::ToolTipText, Qt::white);
    darkPalette.setColor(QPalette::Text, Qt::white);
    darkPalette.setColor(QPalette::Button, QColor(64,64,64));
    darkPalette.setColor(QPalette::ButtonText, Qt::white);
    darkPalette.setColor(QPalette::BrightText, Qt::red);
    darkPalette.setColor(QPalette::Link, QColor(42, 130, 218));

    darkPalette.setColor(QPalette::Disabled, QPalette::Window, QColor(96,96,96));
    darkPalette.setColor(QPalette::Disabled, QPalette::WindowText, Qt::gray);
    darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, Qt::gray);
    darkPalette.setColor(QPalette::Disabled, QPalette::Base, QColor(96,96,96));
    darkPalette.setColor(QPalette::Disabled, QPalette::AlternateBase, QColor(96,96,96));
    darkPalette.setColor(QPalette::Disabled, QPalette::Button, QColor(96,96,96));

    darkPalette.setColor(QPalette::Highlight, QColor(130, 130, 130));
    darkPalette.setColor(QPalette::HighlightedText, Qt::black);

    QApplication::setPalette(darkPalette);

    darkPalette.setColor(QPalette::Window, QColor(0x929292)); //QColor("#929292")
    QApplication::setPalette(darkPalette, "QCheckBox");

    darkPalette.setColor(QPalette::Button, QColor(0x646464));//QColor("#646464")
    QApplication::setPalette(darkPalette, "QToolButton");

    darkPalette.setColor(QPalette::Window, QColor(0x929292));//QColor("#929292")
    darkPalette.setColor(QPalette::Base, QColor(80,80,80));
    QApplication::setPalette(darkPalette, "QLineEdit");

    darkPalette.setColor(QPalette::Mid, Qt::white);
    QApplication::setPalette(darkPalette, "QGroupBox");

    QStringList styleSheetList;
    styleSheetList << QStringLiteral("QToolTip { color: #ffffff; background-color: #929292; border: 1px solid white;}");
    styleSheetList << QStringLiteral("QDockWidget::title {text-align: left center; background: rgb(48, 48, 48);}");
    styleSheetList << QStringLiteral("QDockWidget {titlebar-close-icon: url(:/res/close.png); titlebar-normal-icon: url(:/res/undock.png);}");
    styleSheetList << QStringLiteral("QGroupBox {background-color: transparent; border: 1px solid gray; border-radius: 3px; margin-top: 7px;}");
    styleSheetList << QStringLiteral("QGroupBox::title {subcontrol-origin: margin; top: 0px; left: 15px;}");
    qApp->setStyleSheet(styleSheetList.join(QChar('\n')));

    int drvVer = 0;
    cudaError_t err = cudaSuccess;
    err = cudaDriverGetVersion(&drvVer);
    if(err != cudaSuccess)
    {
        QMessageBox::critical(nullptr, QCoreApplication::applicationName(),
                              QObject::tr("No CUDA driver installed.\nProcessing is impossible."));
        return 0;
    }
    //
    if(drvVer < MIN_DRIVER_VERSION)
    {
        QMessageBox::critical(nullptr, QCoreApplication::applicationName(),
                              QObject::tr("CUDA 10.0 compatible driver required.\nPlease update NVidia drivers to the latest version."));
        return 0;
    }

    int devCount = 0;
    cudaGetDeviceCount(&devCount);

    if(devCount == 0)
    {
        QMessageBox::critical(nullptr, QCoreApplication::applicationName(),
                              QObject::tr("No CUDA device found.\nProcessing is impossible."));
        return 0;
    }

    cudaDeviceProp props{};
    bool found = false;
    for(int i = 0; i < devCount; i++)
    {
        cudaGetDeviceProperties(&props, i);
        if(props.major >= 3)
        {
            found = true;
            break;
        }
    }
    if(!found)
    {
        QMessageBox::critical(nullptr, QCoreApplication::applicationName(),
                              QObject::tr("Kepler architecture or later GPU required.\nProcessing is impossible."));
        return 0;
    }

    MainWindow w;
    w.show();

    return QApplication::exec();
}
