# gpu-camera-sample
Camera sample application with realtime GPU image processing

That software is based on minimal image processing pipeline for camera applications that includes:
* Raw image capture (8-bit, 12-bit packed/unpacked, 16-bit)
* Import to GPU
* Raw data convert and unpack
* Linearization curve
* Bad Pixel Correction  
* Dark frame subtraction  
* Flat-Field Correction
* White Balance
* Exposure Correction (brightness control)
* Debayer with HQLI (5&times;5 window), DFPD (11&times;11), MG (23&times;23) algorithms
* Wavelet-based denoiser
* Gamma (linear, sRGB)
* JPEG / MJPEG encoding
* Output to monitor
* Export from GPU to CPU memory
* Storage of compressed data to SSD

Processing is done on NVIDIA GPU to speedup the performance. The software could also work with raw images in PGM format and you can utilize these images for testing or if you don't have a camera or if your camera is not supported. More info about that project you can find <a href="https://www.fastcompression.com/blog/gpu-software-machine-vision-cameras.htm" target="_blank">here</a>.

From the benchmarks on <strong>NVIDIA GeForce RTX 2080ti</strong> we can see that GPU-based raw image processing is very fast and it could offer very high quality at the same time. The total performane could reach <strong>2 GPix/s</strong> for color cameras and <strong>3 GPix/s</strong> for monochrome cameras. The performance strongly depends on complexity of that pipeline. Multiple GPU solutions could significanly improve the performance.

Currently the software is working with <a href="https://www.ximea.com" target="_blank">XIMEA</a> cameras. Soon we are going to add support for <a href="https://www.jai.com" target="_blank">JAI</a> and <a href="https://www.imperx.com" target="_blank">Imperx</a> cameras. You can add support for desired cameras by yourself. The software is working with demo version of Fastvideo SDK, that is why you can see a watermark on the screen. To get a license for the SDK, please contact <a href="https://www.fastcompression.com/" target="_blank">Fastvideo company</a>.

## How to build gpu-camera-sample

### Requirements for Windows

* Camera SDK for XIMEA, Flir, Baumer, JAI, Imperx, etc. (currently only XIMEA is supported)
* Fastvideo SDK (demo) ver.0.15.0.0
* NVIDIA CUDA-10.1
* Qt ver.5.13.1
* Compiler MSVC 2017

### Build instructions

* ```bash git clone https://github.com/fastvideo/gpu-camera-sample.git ```
* Create OtherLibs folder in project root folder. This folder will contains external libraries, used in gpu-camera-sample application.
* Download Fastvideo SDK from <a href="https://drive.google.com/open?id=1p21TXXC7SCw5PdDVEhayRdMQEN6X11ge">Fastvideo SDK (demo) for Windows-7/10, 64-bit</a> (valid till March 23, 2020), unpack it into <Project root>\OtherLibs\fastvideoSDK folder.
* If you need XIMEA camera support, download XiAPI from https://www.ximea.com/support/documents/4. Install downloaded package (by default into C:\XIMEA). Copy API folder from XIAPI installation folder into <Project root>\OtherLibs folder.
* Open src\GPUCameraSample.pro into Qt Creator.
* By default application will be built with no camera support. The only option is camera simulator based on pgm file. To enable Ximea camera suppoer open common_defs.pri and uncomment line DEFINES += SUPPORT_XIMEA.
* Build project.
* Binaries will be placed into <Project root>\GPUCameraSample_x64 folder.

## Software architecture

gpu-camera-sample is a multithreaded application. It consists of the following threads:

* Main application thread controls application GUI and other threads
* Image acquisition from a camera thread which controls camera data acquisition and CUDA-based image processing thread
* CUDA-based image processing thread. Controls RAW data processing as well as async data writing thread and OpenGL renderer thread.
* OpenGL rendering thread. Renders processed data into OpenGL surface.
* Async data writing thread. Writes processed JPEG data to SSD or streams processed video.

We've implemented the simplest approach for camera application. Camera driver is writing raw data to memory ring buffer, then we copy data from ring buffer to GPU for computations. Full image processing pipeline is done on GPU, so we need just to collect processed frames at the output.

In general case, Fastvideo SDK can import/export data from/to SSD / CPU memory / GPU memory. This is done to ensure compatibility with third-party libraries on CPU and GPU. You can get more info at <a href="https://www.fastcompression.com/download/Fastvideo_SDK_manual.pdf" target="_blank">Fastvideo SDK Manual</a>.

## Using gpu-camera-sample

* Run GPUCameraSample.exe
* Press Open button on the toolbar. This will open the first camera in the system or ask to open PGM file if application was built with no camera support.
* Press Play button. This will start aquiring data from the camera and display it on the screen.
* Adjust zoom with Zoom slider or toggle Fit check box if requires
* Select appropriate output format in the Recording pane (please check that output folder exists in the file system, otherwise nothing will be recorded) and press Record button to start recording to disk. 
* Press Record button again when recording is done.

## Minimum Hardware ans Software Requirements

* Windows-7/10, 64-bit
* The latest NVIDIA driver
* NVIDIA GPU with Kepler architecture, 6xx series minimum
* NVIDIA GPU with 4 GB memory or better
* Intel Core i5 or better
* NVIDIA CUDA-10.1
* Compiler MSVC 2017 (MSVC 2015 is not compatible with CUDA-10.1)

We also recommend to check PCI-Express bandwidth for Host-to-Device and Device-to-Host transfers. For GPU with Gen3 x16 it should be in the range of 10-12 GB/s. GPU memory size could be a bottleneck for high resolution cameras, so please check GPU memory usage in the software.

If you are working with images which reside on HDD, please place them on SSD or M2.

For testing purposes you can utilize the latest NVIDIA GeForce RTX 2060, 2070, 2080ti.

For continuous high performance applications we recommend professional NVIDIA Quadro and Tesla GPUs.

## Roadmap

* GPU pipeline for monochrome cameras - in progress
* H.264/H.265 encoders on GPU - in progress 
* Linux version - in progress
* Resize
* UnSharp Mask
* Rotation to an arbitrary angle
* Support for JAI and Imperx cameras
* JPEG2000 encoder
* Realtime raw compression (lossless and/or lossy)
* Curves and Levels via 1D LUT
* Color correction with 3&times;3 matrix
* Support of other color spaces
* 3D LUT for HSV and RGB
* Defringe module
* DCP support
* LCP support (remap)
* Special version for NVIDIA Jetson hardware and L4T for CUDA-10.0
* Interoparability with external FFmpeg and GStreamer

## Info

  * <a href="https://www.fastcompression.com/product/sdk.htm" target="_blank">Fastvideo SDK for Image & Video Processing</a>

## Fastvideo SDK Benchmarks

* <a href="https://www.fastcompression.com/pub/2019/Fastvideo_SDK_benchmarks.pdf" target="_blank">Fastvideo SDK Benchmarks 2019</a>
* <a href="https://www.fastcompression.com/blog/jetson-benchmark-comparison.htm" target="_blank">Jetson Benchmark Comparison: Nano vs TX1 vs TX2 vs Xavier</a>

## Downloads

* Download <a href="https://www.fastcinemadng.com/download/download.html" target="_blank">Fast CinemaDNG Processor</a> software for Windows, manual and test DNG footages
* Download <a href="https://drive.google.com/open?id=1p21TXXC7SCw5PdDVEhayRdMQEN6X11ge">Fastvideo SDK (demo) for Windows-7/10, 64-bit</a> (valid till March 23, 2020)
* Download <a href="https://drive.google.com/open?id=1GNcQtGmz-FBrKqrsSnMENMCbg44xxWQn">Fastvideo SDK (demo) for Linux Ubuntu 18.04, 64-bit</a> (valid till March 23, 2020)
* Download <a href="https://drive.google.com/file/d/1gBfPkazCiHLHc4piPHSJA2_Rm52CnoKD/view?usp=sharing">Fastvideo SDK (demo) for NVIDIA Jetson Nano, TX2, Xavier</a> (valid till April 12, 2020)
* Download <a href="https://www.fastcompression.com/download/Fastvideo_SDK_manual.pdf" target="_blank">Fastvideo SDK Manual</a>
