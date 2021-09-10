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

#ifndef CUDAPROCESSOROPTIONS_H
#define CUDAPROCESSOROPTIONS_H

#include "fastvideo_sdk.h"
#include "fastvideo_nppGeometry.h"
#include "fastvideo_denoise.h"

#include <memory.h>
#include <QVector>
#include <QString>
#include <QRect>

#include "Globals.h"

class CUDAProcessorOptions
{
public:
    enum VideoCodec
    {
        vcNone = 0,
        vcH264,
        vcMJPG,
        vcJPG,
        vcPGM,
        vcHEVC
    };

    CUDAProcessorOptions()
    {
        Width = 0;
        Height = 0;

        Packed = false;

        Info = false;
        DeviceId = (std::numeric_limits<unsigned>::max)();

        MaxWidth = 0;
        MaxHeight = 0;

        SurfaceFmt = FAST_RGB8;

        BayerType = FAST_MG;
        BayerFormat = FAST_BAYER_BGGR;

        MatrixA = nullptr;
        MatrixB = nullptr;
        EnableSAM = false;

        BlackLevel = 0;
        WhiteLevel = 65535;

        Red = 1.;
        Green = 1.;
        Blue = 1.;
        Temperature = 6504;
        Tint = 0;
        eV = 1.;

        Codec = vcNone;

        ScaleX = 1.0;
        ScaleY = 1.0;

        JpegQuality = 90;
        JpegRestartInterval = 16;
        JpegSamplingFmt = FAST_JPEG_420;
        bitrate = 0;

        EnableDenoise = true;
        Globals::initDenoiseParams(DenoiseStaticParams);
        memset(&DenoiseParams, 0, sizeof(DenoiseParams));
        for(auto & t : DenoiseParams.threshold_per_level)
            t = 1;
        DenoiseParams.enhance[0] = DenoiseParams.enhance[1] = DenoiseParams.enhance[2] = 1.;

        VerFlip = false;
        HorFlip = false;
        Angle = 0;

        ShowPicture = true;

        EnableBPC = false;
    }

    CUDAProcessorOptions(const CUDAProcessorOptions& other)
    {
        Width = other.Width;
        Height = other.Height;

        Packed = other.Packed;

        Info = other.Info;
        DeviceId = other.DeviceId;

        MaxWidth = other.MaxWidth;
        MaxHeight = other.MaxHeight;

        MatrixA = other.MatrixA;
        MatrixB = other.MatrixB;
        EnableSAM = other.EnableSAM;

        SurfaceFmt = other.SurfaceFmt;
        BlackLevel = other.BlackLevel;
        WhiteLevel = other.WhiteLevel;

        BayerType = other.BayerType;
        BayerFormat = other.BayerFormat;

        Codec = other.Codec;

        JpegQuality = other.JpegQuality;
        JpegRestartInterval = other.JpegRestartInterval;
        JpegSamplingFmt = other.JpegSamplingFmt;
        bitrate = other.bitrate;

        Red = other.Red;
        Green = other.Green;
        Blue = other.Blue;
        Temperature = other.Temperature;
        Tint = other.Tint;
        eV = other.eV;

        CropRect = other.CropRect;
        ScaleX = other.ScaleX;
        ScaleY = other.ScaleY;

        VerFlip = other.VerFlip;
        HorFlip = other.HorFlip;
        Angle = other.Angle;

        ShowPicture = other.ShowPicture;

        EnableDenoise = other.EnableDenoise;
        memcpy(&DenoiseParams, &other.DenoiseParams, sizeof(denoise_parameters_t));
        memcpy(&DenoiseStaticParams, &other.DenoiseStaticParams, sizeof(denoise_static_parameters_t));

        EnableBPC = other.EnableBPC;
    }
    ~CUDAProcessorOptions() = default;

    bool isValid()
    {
        return (Width > 0 &&
                Height > 0 &&
                MaxWidth > 0 &&
                MaxHeight > 0 &&
                BayerFormat > FAST_BAYER_NONE);
    }

    unsigned Width;
    unsigned Height;

    bool Info;
    unsigned DeviceId;

    bool ShowPicture;

    bool Packed;

    unsigned int MaxWidth;
    unsigned int MaxHeight;

    fastSurfaceFormat_t SurfaceFmt;
    unsigned short BlackLevel;
    unsigned short WhiteLevel;
    QVector<unsigned short> LinearizationLut;

    fastDebayerType_t BayerType;
    fastBayerPattern_t BayerFormat;

    VideoCodec Codec;

    unsigned JpegQuality;
    unsigned JpegRestartInterval;
    fastJpegFormat_t JpegSamplingFmt;
    int bitrate;

    float Red;
    float Green;
    float Blue;
    float Temperature;
    float Tint;
    float eV;

    float* MatrixA;
    void* MatrixB;
    bool EnableSAM;

    QRect CropRect;

    float ScaleX;
    float ScaleY;

    bool VerFlip;
    bool HorFlip;
    int  Angle;

    bool EnableDenoise;
    denoise_parameters_t DenoiseParams{};
    denoise_static_parameters_t DenoiseStaticParams{};

    bool EnableBPC;

};

#endif // CUDAPROCESSOROPTIONS_H
