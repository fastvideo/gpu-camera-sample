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

#include "AppSettings.h"
#include "QCoreApplication"
#include <QObject>
#include <QDebug>
#include <QDir>
#include "Globals.h"

AppSettingsStorage::AppSettingsStorage():
    QSettings(QSettings::IniFormat, QSettings::UserScope,QCoreApplication::organizationName())
{
}


AppSettings::AppSettings() //: QSettings(QSettings::SystemScope, QCoreApplication::organizationName(), QCoreApplication::applicationName())
{

    AppSettingsStorage settings;
    useSystemICCProfile = settings.value(QStringLiteral("UseSystemICCProfile"), true).toBool();
    customICCProfile = settings.value(QStringLiteral("CustomICCProfile"), QString()).toString();
    useProjectColorSpace = settings.value(QStringLiteral("UseProjectColorSpace"), false).toBool();

    lastLoadImgPath = Globals::completePath(QDir::fromNativeSeparators(settings.value(QStringLiteral("LastLoadImgPath"), QDir::homePath()).toString()));
    rootPath = Globals::completePath(QDir::fromNativeSeparators(settings.value(QStringLiteral("RootPath"), QDir::homePath() + QStringLiteral("/Fastvideo/")).toString()));

    maxOutWidth = settings.value(QStringLiteral("MaxOutWidth"), 2432).toUInt();
    maxOutHeight = settings.value(QStringLiteral("MaxOutHeight"), 1366).toUInt();

    maxInputWidth = settings.value(QStringLiteral("MaxInputWidth"), 0).toUInt();
    maxInputHeight = settings.value(QStringLiteral("MaxInputHeight"), 0).toUInt();

    jpegPixelFormat = fastJpegFormat_t(settings.value(QStringLiteral("JPEG/PixelFormat"), FAST_JPEG_420).toInt());
    if(jpegPixelFormat < FAST_JPEG_444 && jpegPixelFormat > FAST_JPEG_420)
        jpegPixelFormat = FAST_JPEG_420;

    jpegQty = settings.value(QStringLiteral("JPEG/Qty"), 90).toInt();
    jpegQty = qBound<int>(40, jpegQty, 100);


    QString defaultLocaleName(QLocale::system().name());
    defaultLocaleName = QStringLiteral("en_US");
    localeName = settings.value(QStringLiteral("LocaleName"), defaultLocaleName).toString();//

    fixBadPixels = settings.value(QStringLiteral("FixBadPixels"), false).toBool();
}

void AppSettings::save()
{
    AppSettingsStorage settings;
    settings.setValue(QStringLiteral("LastProjectPath"), lastProjectPath);
    settings.setValue(QStringLiteral("LastLoadImgPath"), lastLoadImgPath);
    settings.setValue(QStringLiteral("RootPath"), rootPath);

    settings.setValue(QStringLiteral("MaxOutWidth"), maxOutWidth);
    settings.setValue(QStringLiteral("MaxOutHeight"), maxOutHeight);

    settings.setValue(QStringLiteral("MaxInputWidth"), maxInputWidth);
    settings.setValue(QStringLiteral("MaxInputHeight"), maxInputHeight);

    settings.setValue(QStringLiteral("UseProjectColorSpace"), useProjectColorSpace);
    settings.setValue(QStringLiteral("UseSystemICCProfile"), useSystemICCProfile);
    settings.setValue(QStringLiteral("CustomICCProfile"), customICCProfile);

    settings.setValue(QStringLiteral("JPEG/PixelFormat"), jpegPixelFormat);
    settings.setValue(QStringLiteral("JPEG/Qty"), jpegQty);

    settings.setValue(QStringLiteral("LocaleName"), localeName);

    settings.setValue(QStringLiteral("FixBadPixels"), fixBadPixels);

    settings.sync();
}
