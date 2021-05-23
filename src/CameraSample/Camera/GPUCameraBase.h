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

#ifndef CAMERABASE_H
#define CAMERABASE_H

#include <QObject>
#include <QString>
#include <QMap>
#include <QEvent>
#include <QTimer>
#include <QThread>
#include <QVariant>
#include <unordered_map>

#include "fastvideo_sdk.h"
#include "FrameBuffer.h"

#define  FrameEventID (QEvent::User + 1000)
class FrameEvent : public QEvent
{
public:
    explicit FrameEvent():QEvent(static_cast<QEvent::Type>(FrameEventID)){}
};

class RawProcessor;

///Base class for camera object
class GPUCameraBase : public QObject
{
    Q_OBJECT
public:

    typedef enum{
        cif8bpp = 0, ///8 bit per pixel
        cif10bpp,    ///10 bit per pixel in 16 bit
        cif12bpp,    ///12 bit per pixel in 16 bit
        cif12bpp_p,  ///12 bit per pixel packed (2 pixel in 3 bytes)
        cif16bpp     ///16 bit per pixel
    } cmrImageFormat;

    typedef enum{
        cstClosed = 0,
        cstStopped,
        cstStreaming
    } cmrCameraState;

    typedef enum{
        prmFrameRate = 0,
        prmExposureTime,
        prmLast
    } cmrCameraParameter;

    enum class cmrCameraStatistic {
        statFramesTotal = 0, /// Total number of frames acquired
        statFramesDropped ,  /// Number of dropped frames
        statFramesIncomplete , /// Number of incomplete frames
        statCurrFrameID,    /// Current Frame ID (blockID)
        statCurrTimestamp,  /// Current Frame Timestamp
        statCurrTroughputMbs100, /// Average Thoughtput in Megabits per 100 seconds
        statCurrFps100 /// FPS multiplied by 100
    } ;

    struct cmrParameterInfo{
        cmrCameraParameter param;
        float min;
        float max;
        float increment;

        cmrParameterInfo(){}
        cmrParameterInfo(cmrCameraParameter p){param = p;}
    } ;

    explicit GPUCameraBase();

    ///Connect to camera, initialize internal variables and
    ///allocate required resources
    virtual bool open(uint32_t devID) = 0;

    ///Start streaming frames from camera
    virtual bool start() = 0;

    ///Stop streaming frames from camera
    virtual bool stop() = 0;

    ///Disconnect from camera and free allocated resources
    virtual void close() = 0;

    ///Get camera parameter value. Return true on success, false otherwise.
    virtual bool getParameter(cmrCameraParameter param, float& val) = 0;

    ///Set parameter value. Return true on success, false otherwise.
    virtual bool setParameter(cmrCameraParameter param, float val) = 0;

    ///Get camera parameter information. Return true on success, false otherwise.
    virtual bool getParameterInfo(cmrParameterInfo& info) = 0;


    ///Get current camera state
    cmrCameraState state() {return mState;}

    ///Camera bayer pattern
    fastBayerPattern_t bayerPattern(){return mPattern;}

    ///Camera FPS
    float fps(){return mFPS;}

    ///Return vendor specific device ID
    uint32_t devID(){return mDevID;}

    ///Return FV SDK compatible surface format
    fastSurfaceFormat_t surfaceFormat(){return mSurfaceFormat;}


    bool isPacked(){return mImageFormat == cif12bpp_p;}
    bool isColor(){return mIsColor;}

    int width() {return mWidth;}
    int height(){return mHeight;}

    int whiteLevel(){return mWhite;}
    int blackLevel(){return mBblack;}

    ///Return camera info
    QString model(){return mModel;}
    QString manufacturer(){return mManufacturer;}
    QString serial(){return mSerial;}

    CircularBuffer* getFrameBuffer(){return &mInputBuffer;}

    void setProcessor(RawProcessor* proc){QMutexLocker l(&mLock); mRawProc = proc;}

    /// Get camera statistics
    bool  GetStatistics(cmrCameraStatistic stat, uint64_t &outStat) {
        if (mStatistics.find(stat)==mStatistics.end())
            return false;
        outStat = mStatistics[stat];
        return true;
    };

signals:
    void stateChanged(GPUCameraBase::cmrCameraState newState);

protected:
    QString mModel;
    QString mManufacturer;
    QString mSerial;

    int mWidth  = 0;
    int mHeight = 0;

    float mFPS  = 0;

    int mWhite  = 255;
    int mBblack = 0;

    uint32_t mDevID = 0;

    bool                mIsColor = true;
    cmrCameraState      mState = cstClosed;
    fastBayerPattern_t  mPattern = FAST_BAYER_NONE;
    fastSurfaceFormat_t mSurfaceFormat = FAST_I8;
    cmrImageFormat      mImageFormat = cif8bpp;
    bool                mStreaming = false;
    CircularBuffer      mInputBuffer;
    RawProcessor*       mRawProc = nullptr;

    QThread mCameraThread;
    QMutex  mLock;

    std::unordered_map<cmrCameraStatistic, uint64_t> mStatistics;

    uint64_t mCurrFrameID{0};
    uint64_t mDropFramesNum{0};
    std::chrono::time_point<std::chrono::system_clock> mFirstFrameTime;
    std::chrono::time_point<std::chrono::system_clock> mPrevFrameTime;
    const std::chrono::time_point<std::chrono::system_clock> kZeroTime;
    uint64_t mTotalBytesTransferred{0};
};

///Base class for camera enumeration
class CameraEnumBase
{
public:
    CameraEnumBase();

    ///Enumertes connected cameras
    virtual QMap<uint32_t, QString> enumCameras() = 0;

private:

};

#endif // CAMERABASE_H
