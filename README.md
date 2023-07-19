# gpu-camera-sample
Camera sample application with realtime GPU image processing performance (Windows, Linux, Jetson)

<p><a target="_blank" href="https://www.fastcompression.com/blog/gpu-software-machine-vision-cameras.htm">
<img src="https://www.fastcompression.com/img/blog/machine-vision/gpu-software-machine-vision-cameras.png" alt="gpu software machine vision genicam" style="max-width:100%"/></a></p>

That software is based on the following image processing pipeline for camera applications that includes:
* Raw image capture (monochrome or bayer 8-bit, 12-bit packed/unpacked, 16-bit)
* Import to GPU
* Raw data conversion and unpacking
* Linearization curve
* Bad Pixel Correction 
* Dark frame subtraction 
* Flat-Field Correction
* White Balance
* Exposure Correction (brightness control)
* Debayer with HQLI (5&times;5 window), L7 (7&times;7 window), DFPD (11&times;11), MG (23&times;23) algorithms
* Color correction with 3&times;3 matrix
* Wavelet-based denoiser
* Crop / Resize / Flip / Flop / Rotate
* Gamma (linear, sRGB)
* JPEG / MJPEG encoding/decoding
* H.264 and HEVC encoding/decoding
* Output to monitor via OpenGL
* Export from GPU to CPU memory
* MJPEG and H.264/H.265 streaming
* Storage of compressed images/video to SSD

Processing is done on NVIDIA GPU to speedup the performance. The software could also work with raw bayer images in PGM format and you can utilize these images for testing or if you don't have a camera, or if your camera is not supported. More info about that project you can find <a href="https://www.fastcompression.com/blog/gpu-software-machine-vision-cameras.htm" target="_blank">here</a>.

From the benchmarks on <strong>NVIDIA Quadro RTX 6000</strong> or <strong>GeForce RTX 2080ti</strong> we can see that GPU-based raw image processing is very fast and it could offer high image quality at the same time. The total performance could reach <strong>4 GPix/s</strong> for color cameras. The performance strongly depends on complexity of the pipeline. Multiple GPU solutions could significantly improve the performance.

Currently the software is working with <a href="https://www.ximea.com" target="_blank">XIMEA</a> cameras via XIMEA SDK. <a href="https://www.flir.com" target="_blank">FLIR</a> cameras are supported via Spinnaker SDK. We can work with <a href="https://www.imperx.com" target="_blank">Imperx</a> cameras via Imperx SDK. <a href="https://thinklucid.com" target="_blank">LUCID Vision Labs</a> cameras are supported via Arena SDK.

Via GenICam the software can work with <a href="https://www.ximea.com" target="_blank">XIMEA</a>, <a href="https://www.matrix-vision.com" target="_blank">MATRIX VISION</a>, <a href="https://www.baslerweb.com" target="_blank">Basler</a>, <a href="https://www.flir.com" target="_blank">FLIR</a>, <a href="https://www.imperx.com" target="_blank">Imperx</a>, <a href="https://www.jai.com" target="_blank">JAI</a>, <a href="https://thinklucid.com" target="_blank">LUCID Vision Labs</a>, <a href="https://dahengimaging.com/" target="_blank">Daheng Imaging</a> cameras.

The software is also working with <a href="https://www.leopardimaging.com" target="_blank">Leopard Imaging</a> mipi csi cameras on Jetson. You need to have a proper driver to be able to acquire raw frames from mipi camera for further image processing on the GPU with 16/32-bit precision. The software doesn't use NVIDIA ISP via libargus.

Soon we are going to add support for <a href="https://emergentvisiontec.com/" target="_blank">Emergent Vision Technologies</a>, <a href="https://en.ids-imaging.com/" target="_blank">IDS Imaging Development Systems</a>, <a href="https://www.baumer.com" target="_blank">Baumer</a>, <a href="https://kayainstruments.com/" target="_blank">Kaya Instruments</a> cameras. You can add support for desired cameras by yourself. The software is working with demo version of <a href="https://www.fastcompression.com/products/sdk.htm" target="_blank">Fastvideo SDK</a>, that is why you can see a watermark on the screen. To get a Fastvideo SDK license for development and for deployment, please contact <a href="https://www.fastcompression.com/" target="_blank">Fastvideo company</a>.

## How to build gpu-camera-sample

### Requirements for Windows

* Camera SDK or GenICam package + camera vendor GenTL producer (.cti). Сurrently XIMEA, MATRIX VISION, Basler, FLIR, Imperx, JAI, LUCID Vision Labs, Daheng Imaging cameras are supported
* Fastvideo SDK (demo) ver.0.17.6.1
* NVIDIA CUDA-12.1
* Qt ver.5.13.1
* Compiler MSVC 2022 or later

### Requirements for Linux

* Ubuntu 22.04 for x64 platform, Ubuntu 20.04 for Arm64 platform with CUDA 11, Ubuntu 18.04 for Arm64 platform with CUDA 10
* Camera SDK or GenICam package + camera vendor GenTL producer (.cti). Currently XIMEA, MATRIX VISION, Basler, FLIR, Imperx, JAI, Daheng Imaging cameras are supported
* Fastvideo SDK (demo) ver.0.18.1.0
* NVIDIA CUDA-12.1 for x64, CUDA-11.4 (Jetson AGX Xavier, Orin) or CUDA-10.2 (Jetson Tx2, NX) for ARM64 platform 
* Compiler gcc 7.4 or later
* Qt 5 (qtbase5-dev)
``` console
sudo apt-get install qtbase5-dev qtbase5-dev-tools qtcreator git
```
* FFmpeg libraries
``` console 
sudo apt-get install  libavutil-dev libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavresample-dev libx264-dev
```
Jetson users have to build FFmpeg libraries from sources. See [this shell script](Scripts/build_ffmpeg.sh) for details.
* Libjpeg and zlib libraries
``` console 
sudo apt-get install libjpeg-dev zlib1g-dev
```

### Build instructions

* Obtain source code: 
``` console
git clone https://github.com/fastvideo/gpu-camera-sample.git 
```

### For Windows users

* Create OtherLibs folder in the project root folder. This folder will contains external libraries, used in gpu-camera-sample application.
* Download Fastvideo SDK from <a href="https://drive.google.com/file/d/1e4lMN1gOOL9M8zPI_tFHCHM5sWFKL4YE/view?usp=sharing">Fastvideo SDK (demo) for Windows-7/10, 64-bit</a>, unpack it into \<ProjectRoot\>/OtherLibs/FastvideoSDK folder. If the trial period is expired, please send an inquiry to Fastvideo to obtain the latest version.
* If you need direct XIMEA camera support, download XiAPI from https://www.ximea.com/support/wiki/apis/XIMEA_Windows_Software_Package. Install downloaded package (by default into C:\XIMEA). Copy API folder from XIAPI installation folder into \<ProjectRoot\>/OtherLibs folder.
* To work with FLIR cameras
   * Download Spinnaker SDK from https://www.flir.com/support-center/iis/machine-vision/downloads/spinnaker-sdk-and-firmware-download/. 
   * Install downloaded package.
   * Copy bin64, include and lib64 folders from c:/Program Files/FLIR System/Spinnaker to \<ProjectRoot\>/OtherLibs/FLIR
* To work with Imperx cameras, download Imperx SDK (IpxCameraSdk) from https://www.imperx.com/downloads/. Unpack and copy archive content to \<ProjectRoot\>/OtherLibs/Imperx folder.
* To work with LUCID Vision Labs cameras, download Arena SDK from https://thinklucid.com/downloads-hub/. Install it and copy "C:\Program Files\Lucid Vision Labs\Arena SDK" folder into \<ProjectRoot\>/OtherLibs.
* If you need GenICam support
   * Download GenICamTM Package Version 2019.11 from https://www.emva.org/wp-content/uploads/GenICam_Package_2019.11.zip
   * If you are going to use Imperx cameras, download GenICamTM Package Version 3.0.2 from https://www.emva.org/wp-content/uploads/GenICam_Package_v3_0_2.zip
   * Unpack it to a temporary folder and cd to Reference Implementation folder.
   * Create \<ProjectRoot\>/OtherLibs/GenICam folder.
   * Unpack GenICam_V3_2_0-Win64_x64_VC141-Release-SDK.zip into \<ProjectRoot\>/OtherLibs/GenICam folder.
   * Unpack GenICam_V3_2_0-Win64_x64_VC141-Release-Runtime.zip into \<ProjectRoot\>/OtherLibs/GenICam/library/CPP

You also can download precompiled libs from <a href="https://drive.google.com/file/d/1NTcWsZdT1fhSlf6yljMNpSxAwyTIlCYZ/view?usp=drive_link" target="_blank">here</a>
* By default the application will be built with no camera support. The only option is camera simulator which is working with PGM files. 
* Open \<ProjectRoot\>/src/GPUCameraSample.pro in Qt Creator.
* Open common_defs.pri
* To enable GenICam support, uncomment DEFINES += SUPPORT_GENICAM
* To enable XIMEA camera support, uncomment DEFINES += SUPPORT_XIMEA
* To enable Lucid Vision Labs camera support, uncomment DEFINES += SUPPORT_LUCID
* To enable FLIR camera support, uncomment DEFINES += SUPPORT_FLIR
* To enable Imperx camera support, uncomment DEFINES += SUPPORT_IMPERX.
* Build the project
* Binaries will be placed into \<ProjectRoot\>/GPUCameraSample_x64 folder.

### For Linux users

Here and after we assume you put source code into home directory, so project root is ~/gpu-camera-sample
* Make sure file \<ProjectRoot\>/Scripts/make_links.sh is executable
``` console
chmod 755 ~/gpu-camera-sample/Scripts/make_links.sh
```
* Create OtherLibsLinux folder in the project root folder. This folder will contain external libraries, used in gpu-camera-sample application.
* Download Fastvideo SDK x64 platform from <a href="https://drive.google.com/file/d/14YwjKvooRos1yKFzlRtdKr5RfS_vUnnZ/view?usp=sharing">Fastvideo SDK (demo) for Linux Ubuntu 18.04, 64-bit</a>, or Fastvideo SDK Arm64 platform from <a href="https://drive.google.com/file/d/1HgfqboijA8VlQAulEm69a7ayx-hQVxuO/view?usp=sharing">Fastvideo SDK (demo) for NVIDIA Jetson Nano, TX2, Xavier</a> and unpack it into \<ProjectRoot\>/OtherLibsLinux/FastvideoSDK folder. Copy all files from \<ProjectRoot\>/OtherLibsLinux/FastvideoSDK/fastvideo_sdk/lib to \<ProjectRoot\>/OtherLibsLinux/FastvideoSDK/fastvideo_sdk/lib/Linux64 for x64 platform and to \<ProjectRoot\>/OtherLibsLinux/FastvideoSDK/fastvideo_sdk/lib/Arm64 for Arm64 platform.
* Create links to Fastvideo SDK *.so files
``` console
cd ~/gpu-camera-sample/Scripts
./make_links.sh ~/gpu-camera-sample/OtherLibsLinux/FastvideoSDK/fastvideo_sdk/lib/Linux64
```
* If you need direct XIMEA camera support, download XiAPI from https://www.ximea.com/support/wiki/apis/XIMEA_Linux_Software_Package. Unpack and install downloaded package.
* If you need GenICam support
   * Download GenICamTM Package Version 2019.11 (https://www.emva.org/wp-content/uploads/GenICam_Package_2019.11.zip).
   * Unpack it to a temporary folder and cd to Reference Implementation folder.
   * Create \<ProjectRoot\>/OtherLibsLinux/GenICam folder.
   * Unpack GenICam_V3_2_0-Linux64_x64_gcc48-Runtime.tgz or GenICam_V3_2_0-Linux64_ARM_gcc49-Runtime.tgz into \<ProjectRoot\>/OtherLibsLinux/GenICam folder.
   * Unpack GenICam_V3_2_0-Linux64_x64_gcc48-SDK.tgz or GenICam_V3_2_0-Linux64_ARM_gcc49-SDK.tgz into \<ProjectRoot\>/OtherLibsLinux/GenICam/library/CPP
   * Create  \<ProjectRoot\>OtherLibsLinux\GenICam\library\CPP\include\GenTL folder and copy there
      * GenICam_Package_2019.11.zip\GenICam_Package_2019.11\GenTL\GenTL_v1_6.zip\GenTL.h
      * GenICam_Package_2019.11.zip\GenICam_Package_2019.11\SFNC\PFNC.h
   * Ensure Qt uses gcc, not clang to build project.
* If you need LUCID Vision Labs cameras support.
    * Download Arena SDK for Linux from https://thinklucid.com/downloads-hub/ and unpack it into OtherLibsLinux/Arena SDK/ArenaSDK_Linux_x64/ for x64 platform and OtherLibsLinux/Arena SDK/ArenaSDK_Linux_ARM64 for arm64 platform.
    * Run
``` console
        sudo /.Arena_SDK_Linux_x64.conf
```
for x64 platform and
``` console
        sudo /.Arena_SDK_ARM64.conf
```
for arm64 platform
cd to precompiledExamples and run
``` console
        ./IpConfigUtility /list
```
if everything is OK you will see something like that
``` console
    Scanning for devices...
    index MAC             IP              SUBNET          GATEWAY                 IP CONFIG
    0     1C0FAF5A908A    169.254.139.144 255.255.0.0     0.0.0.0                 DHCP= 1 Persistent Ip= 0 LLA = 1
```
run
``` console
        ./Cpp_Acquisition
```
to test that camera is working.
* If you need built-in MIPI cameras support on Jetson TX2 or Leopard Imaging IMX477 MIPI SCI camera support on Jetson AGX Xavier, read  [this guide](LI-IMX477-MIPI.md)  how to setup units and install required drivers.
* By default the application will be built with no camera support. The only option is camera simulator which is working with PGM files. 
* Open \<ProjectRoot\>/src/GPUCameraSample.pro in Qt Creator.
* Open common_defs.pri
* To enable GenICam support, uncomment DEFINES += SUPPORT_GENICAM. 
* To enable XIMEA camera support, uncomment DEFINES += SUPPORT_XIMEA
* To enable Imperx camera support, uncomment DEFINES += SUPPORT_IMPERX.
* To enable Lucid Vision Labs camera support, uncomment DEFINES += SUPPORT_LUCID
* FLIR support is experimental at the moment. Use it on your own risk.
* Build the project.
* If GenICam support is enabled, set environment variable GENICAM_GENTL64_PATH with full path to the camera vendor GenTL producer (.cti) library, before run the application.
* Binaries will be placed into \<ProjectRoot\>/GPUCameraSample_Arm64 or GPUCameraSample_Linux64 folder. To run the application from the terminal run GPUCameraSample.sh. Necessary symbolic links will be made during compile time.

You also can download precompiled libs from <a href="https://drive.google.com/file/d/1h5EeCLHjmDxBSKo5usXbATOYXlurrRO1/view?usp=drive_link" target="_blank">here</a>

### How to work with NVIDIA Jetson to get maximum performance

NVIDIA Jetson provides many features related to power management, thermal management, and electrical management. These features deliver the best user experience possible given the constraints of a particular platform. The target user experience ensures the perception that the device provides:

* Uniformly high performance
* Excellent battery life
* Perfect stability

Utility <strong>nvpmodel</strong> has to been used to change the power mode. Mode with power consumption is MAXN. To activate this mode call 
``` console
sudo /usr/sbin/nvpmodel –m 0
```

Also you have to call <strong>jetson_clocks</strong> script to maximize Jetson performance by setting the static maximum frequencies of the CPU, GPU, and EMC clocks. You can also use the script to show current clock settings, store current clock settings into a file, and restore clock settings from a file.
``` console
sudo /usr/bin/jetson_clocks
```

NVIDIA Jetson TX2 has two CPU core types. These are Denver2 and A57. During benchmarking of <a href="https://www.fastcompression.com/products/sdk.htm" target="_blank">Fastvideo SDK</a> we have realized that better performance for J2K encoder and <a href="https://www.fastcompression.com/blog/j2k-codec-on-jetson-tx2.htm" target="_blank">decoder</a> could be achieved with A57 core type. Affinity mask has to be assigned to run only on A57 cores. Linux command taskset assign process affinity. 
``` console
taskset -c 3,4,5 myprogram
```

TX2 has the following core numbers: 0 – A57; 1, 2 – Denver2; 3, 4, 5 – A57. Core 0 is used by Linux for interrupt processing. We do not recommend include it in the user affinity mask. 

## Glass-to-Glass Time Measurements
To check system latency we've implemented the software to run G2G tests in the gpu-camera-sample application. 

We have the following choices for G2G tests:
* Camera captures frame with current time from high resolution timer at the monitor, we send data from camera to the software, do image processing on GPU and then show processed image at the same monitor close to the window with the timer. If we stop the software, we see two different times and their difference is system latency.
* We have implemented more complicated solution: after image processing on GPU we've done JPEG encoding (MJPEG on CPU or on GPU), then send MJPEG stream to receiver process, where we do MJPEG parsing and decoding, then frame output to the monitor. Both processes (sender and receiver) are running at the same PC.
* The same solution as in the previous approach, but with H.264 encoding/decoding (CPU or GPU), both processes are at the same PC.

We can also measure the latency for the case when we stream compressed data from one PC to another over network. Latency depends on camera frame rate, monitor fps, NVIDIA GPU performance, network bandwidth, complexity of image processing pipeline, etc.

## Software architecture

gpu-camera-sample is a multithreaded application. It consists of the following threads:

* Main application thread to control app GUI and other threads.
* Image acquisition from a camera thread which controls camera data acquisition and CUDA-based image processing thread.
* CUDA-based image processing thread. Controls RAW data processing, async data writing thread, and OpenGL renderer thread.
* OpenGL rendering thread. Renders processed data into OpenGL surface.
* Async data writing thread. Writes processed JPEG/MJPEG data to SSD or streams processed video.

Here we've implemented the simplest approach for camera application. Camera driver is writing raw data to memory ring buffer, then we copy data from that ring buffer to GPU for computations. Full image processing pipeline is done on GPU, so we need just to collect processed frames at the output.

In general case, Fastvideo SDK can import/export data from/to SSD / CPU memory / GPU memory. This is done to ensure compatibility with third-party libraries on CPU and GPU. You can get more info at <a href="https://www.fastcompression.com/download/Fastvideo_SDK_manual.pdf" target="_blank">Fastvideo SDK Manual</a>.

## Using gpu-camera-sample

* Run GPUCameraSample.exe
* Press Open button on the toolbar. This will open the first camera in the system or ask to open PGM file (bayer or grayscale) if application was built with no camera support.
* Press Play button. This will start data acquisition from the camera and display it on the screen.
* Adjust zoom with Zoom slider or toggle Fit check box if requires.
* Select appropriate output format in the Recording pane (please check that output folder exists in the file system, otherwise nothing will be recorded) and press Record button to start recording to disk.
* Press Record button again to stop the recording.

## Minimum Hardware ans Software Requirements for desktop application

* Windows-10, Ubuntu 18.04 64-bit
* The latest NVIDIA driver
* NVIDIA GPU with Maxwell architecture, 9xx series minimum
* NVIDIA GPU with 4-8-12 GB memory or better
* Intel Core i5 or better
* NVIDIA CUDA-11.4
* Compiler MSVC 2019 for Windows or gcc 7.4.0 for Linux

We also recommend to check PCI-Express bandwidth for Host-to-Device and Device-to-Host transfers. For GPU with Gen3 x16 it should be in the range of 10-12 GB/s. GPU memory size could be a bottleneck for image processing from high resolution cameras, so please check GPU memory usage in the software.

If you are working with images which reside on HDD, please place them on SSD or M2.

For testing purposes you can utilize the latest NVIDIA GeForce RTX 2060/2070/2080ti, 3070/3080ti/3090, 4080/4090 or Jetson TX2, NX and AGX Xavier.

For continuous high performance applications we recommend professional NVIDIA Quadro and Tesla GPUs.

## Multi-camera applications

To run the software for multi-camera setups, we recommend to run one process per camera. If you have enough GPU memory and processing performance is ok, this is the simplest solution, which was tested in many applications. This is also a good choice for Linux solutions, please don't forget to turn on CUDA MPS.

You can also create a software module to collect frames from different cameras and process them at the same pipeline with gpu-camera-sample application. In that case you will need less GPU memory which could be important for embedded solutions.

Please bear in mind that this is just a <strong>sample application</strong>. It's intended to show how machine vision cameras can work with Fastvideo SDK to get high perfromance image processing on the NVIDIA GPU. 

To test a real application with XIMEA cameras (USB3 or PCIe), please have a look at the following page and download <a href="https://www.fastcompression.com/products/fastvcr-ximea-software.htm" target="_blank">FastVCR software</a>. That software with GenICam (GenTL) support will be released soon.

## Roadmap

* GPU pipeline for monochrome cameras - done
* GenICam Standard support - done
* Linux version - done
* Software for NVIDIA Jetson hardware and L4T for CUDA-10.2 (Jetson Nano, TX2, Xavier AGX and NX) - done
* Glass-to-Glass (G2G) test for latency measurements - done
* Support for XIMEA, MATRIX VISION, Basler, FLIR, Imperx, JAI, Daheng Imaging cameras - done
* MJPEG and H.264 streaming with or without FFmpeg RTSP - done
* HEVC (H.265) encoder/decoder - done
* <a href="https://imaginghub.com/projects/455-real-time-image-processing-on-nvidia-gpu-with-basler-pylon-and-fastvideo" target="_blank">Real-time Image Processing on NVIDIA GPU with Basler pylon</a> - done
* Benchmarks for Jetson Xavier NX - done
* CUDA-11.4 support - done
* Support for MIPI CSI-2 camera which is embedded into TX2 - done
* Support for LUCID Vision Labs cameras - done
* Support for Leopard Imaging MIPI CSI-2 camera with IMX477 image sensor (4K/60fps) - done
* Defringe - done
* <a href="https://www.fastcompression.com/products/fastvcr-ximea-software.htm" target="_blank">FastVCR software for XIMEA cameras</a> - done
* Support for Jetson AGX Orin and CUDA-11.4 - done
* HDR image processing on GPU for automotive 16/20/24-bit image sensors (IMX490, IMX728, AR0820, OX08, etc.) - done
* High performance JPEG-XS decoder on GPU - done
* Support for XIMEA MU181CR-ON camera - done
* Support for Jetson AGX Orin (CUDA-11.4) and CUDA-11.7 - done
* Fast undistortion on GPU with precise and compact maps - in progress
* Support for Jetson Orin (CUDA-11.8) and CUDA-12.1 - in progress
* High performance chromatic aberration suppression in RAW domain - in progress
* Support for Emergent Vision Technologies, DALSA, Baumer, Kaya Instruments, SVS-Vistek cameras - in progress
* RAW Bayer codec
* JPEG2000 encoder and decoder on GPU for camera applications
* Interoperability with FFmpeg, UltraGrid, and GStreamer

## Info

* <a href="https://www.fastcompression.com/products/sdk.htm" target="_blank">Fastvideo SDK for Image & Video Processing on GPU</a>
* <a href="https://www.fastcinemadng.com/" target="_blank">Full image processing pipeline on GPU for digital cinema applications</a>
* <a href="https://en.wikipedia.org/wiki/Nvidia_NVENC#Versions" target="_blank">Parameters and restrictions of NVIDIA NVENC</a> 
* <a href="https://en.wikipedia.org/wiki/Nvidia_NVDEC#GPU_support" target="_blank">Parameters and restrictions of NVIDIA NVDEC</a>
* <a href="https://www.fastcompression.com/blog/content.htm" target="_blank">Fastvideo Blog</a>
* <a href="https://www.fastcompression.com/blog/fastvideo-sdk-vs-nvidia-npp.htm" target="_blank">Fastvideo SDK vs NVIDIA NPP Library</a>
* <a href="https://www.fastcompression.com/blog/gpu-vs-cpu-fast-image-processing.htm" target="_blank">GPU vs CPU at Image Processing. Why GPU is much faster than CPU?</a>
* <a href="https://www.fastcompression.com/blog/jetson-image-processing-framework.htm" target="_blank">Image Processing Framework on Jetson</a>

## Fastvideo SDK Benchmarks

* <a href="https://www.fastcompression.com/pub/2022/Fastvideo_SDK_benchmarks.pdf" target="_blank">Fastvideo SDK Benchmarks</a>
* <a href="https://www.fastcompression.com/blog/jetson-benchmark-comparison.htm" target="_blank">Jetson Benchmark Comparison for Image Processing: TX2 vs Xavier NX vs Xavier AGX, vs Orin AGX</a>
* JPEG2000 benchmarks for <a href="https://www.fastcompression.com/benchmarks/benchmarks-j2k.htm" target="_blank">encoding</a> and <a href="https://www.fastcompression.com/benchmarks/decoder-benchmarks-j2k.htm" target="_blank">decoding</a>

## Downloads

* Download <a href="https://www.fastcinemadng.com/download/download.html" target="_blank">Fast CinemaDNG Processor</a> software for Windows, manual and test DNG and BRAW footages
* FastVCR software for XIMEA cameras with GPU-based image processing - <a href="https://www.fastcompression.com/download/FastVCR_Portable.7z">download link for Windows version</a> 
* Download <a href=https://drive.google.com/file/d/1s56wzH3xZg9lrXW-w1NyrHcmlZpoZzTW/view?usp=sharing" target="_blank">Fastvideo SDK (demo) for Windows-10, 64-bit</a> (valid till June 15, 2024)
* Download <a href="https://drive.google.com/file/d/1ezAPWKz_ovLsmQiID-mlVIvRD2p0FQKI/view?usp=sharing" target="_blank">Fastvideo SDK (demo) for Linux Ubuntu 22.04, 64-bit</a> (valid till June 30, 2024)
* Download <a href="https://drive.google.com/file/d/1TgB1A0Yz8BEHZKsgo9d80ntEJd0hLjsf/view?usp=sharing" target="_blank">Fastvideo SDK (demo) for NVIDIA Jetson Nano, TX2, NX</a> (valid till June 25, 2024)
* Download <a href="https://drive.google.com/file/d/1jw9QfdKs4nw18ZwIiNTjwifjWyDnyzwu/view?usp=sharing" target="_blank">Fastvideo SDK (demo) for NVIDIA Jetson Xavier, Orin</a> (valid till July 06, 2024)
* Download <a href="https://www.fastcompression.com/download/Fastvideo_SDK_manual.pdf" target="_blank">Fastvideo SDK Manual</a>
* <a href="https://imaginghub.com/projects/455-real-time-image-processing-on-nvidia-gpu-with-basler-pylon-and-fastvideo" target="_blank">Real-time Image Processing on NVIDIA GPU with Basler pylon</a>
