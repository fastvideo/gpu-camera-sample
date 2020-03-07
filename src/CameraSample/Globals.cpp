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

#include "Globals.h"
#include <QFileInfo>
#include <QDir>
#include <QApplication>


bool Globals::gEnableLog = false;

qint64  Globals::MaxFileSize = 256*1024*1024;


Globals::Globals() = default;

Globals::~Globals() = default;



QString Globals::getUnicFileName(const QString& fileName)
{
    QFileInfo fi(fileName);
    QString path = fi.path();
    QString name = fi.baseName();
    QString ext = fi.suffix();

    QString ret = fileName;

    int i = 1;
    while(QFileInfo::exists(ret) || i < 1000)
    {
        ret = QStringLiteral("%1/%2_%3.%4").arg(path,name).arg(i).arg(ext);
        i++;
    }
    return (i >= 1000) ? QString() : ret;
}

QString Globals::completePath(const QString& path)
{
    if(path.isEmpty())
        return QString();
    return path.endsWith(QChar('/')) ? path : path + QChar('/');
}

void Globals::initDenoiseParams(denoise_static_parameters_t& params)
{
    memset(&params, 0, sizeof(params));
    params.function = FAST_THRESHOLD_FUNCTION_SOFT;
    params.wavelet = FAST_WAVELET_CDF97;
}

void Globals::validateDenoiseParams(denoise_parameters_t& params)
{
    if(params.threshold[0] < 0)
        params.threshold[0] = 0;

    if(params.threshold[1] < 0)
        params.threshold[1] = 0;

    if(params.threshold[2] < 0)
        params.threshold[2] = 0;

    params.enhance[0] = qBound<float>(0, params.enhance[0], 2.);
    params.enhance[1] = qBound<float>(0, params.enhance[1], 2.);
    params.enhance[2] = qBound<float>(0, params.enhance[2], 2.);

    if(params.dwt_levels > 11)
        params.dwt_levels = 7;

    if(params.dwt_levels < 1)
        params.dwt_levels = 7;
}

void Globals::validateStaticDenoiseParams(denoise_static_parameters_t& params)
{

    if(params.function <= FAST_THRESHOLD_FUNCTION_UNKNOWN)
        params.function = FAST_THRESHOLD_FUNCTION_SOFT;

    if(params.function > FAST_THRESHOLD_FUNCTION_GARROTE)
        params.function = FAST_THRESHOLD_FUNCTION_SOFT;

    if(params.wavelet <= FAST_WAVELET_CDF97)
        params.wavelet = FAST_WAVELET_CDF97;
    if(params.wavelet > FAST_WAVELET_CDF53)
        params.wavelet = FAST_WAVELET_CDF97;
}
