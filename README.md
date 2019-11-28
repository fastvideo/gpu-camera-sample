# gpu-camera-sample
Camera sample application with realtime GPU processing

<p>That software is based on standard image processing pipeline for camera applications. Processing is done on NVIDIA GPU to speedup the performance. The software could also work with raw images in PGM format and you can utilize these images for testing or if you don't have a camera or if your camera is not supported. More info about that project you can find <a href="https://www.fastcompression.com/blog/gpu-software-machine-vision-cameras.htm" target="_blank">here</a>.</p>

<p>From the benchmarks on <strong>NVIDIA GeForce RTX 2080ti</strong> we can see that GPU-based raw image processing is very fast and it could offer very high quality at the same time. The total performane could reach <strong>2 GPix/s</strong> for color cameras and <strong>3 GPix/s</strong> for monochrome cameras. The performance strongly depends on complexity of that pipeline. Multiple GPU solutions could significanly improve the performance.</p>

<p>Currently the software is working with <a href="https://www.ximea.com" target="_blank">XIMEA</a> cameras. Soon we are going to add support for JAI and Imperx cameras. You can add support for desired cameras by yourself. The software is working with demo version of Fastvideo SDK, that is why you can see a watermark on the screen. To get a license for the SDK, please contact <a href="https://www.fastcompression.com/"  target="_blank">Fastvideo company</a>.</p>

<h2>The software has the following architecture</h2>
<ul>
  <li>Thread for GUI and visualization (app main thread)</li>
  <li>Thread for image acquisition from a camera</li>
  <li>Thread to control CUDA-based image processing</li>
  <li>Thread for OpenGL rendering</li>
  <li>Thread for async data writing to SSD or streaming</li>
</ul>
<p>We've implemented the simplest approach for camera application. Camera driver is writing raw data to memory ring buffer, then we copy data from ring buffer to GPU for computations. Full image processing pipeline is done on GPU, so we need just to collect processed frames at the output.</p>
<p>In general case, Fastvideo SDK can import/export data from/to SSD / CPU memory / GPU memory. This is done to ensure compatibility with third-party libraries on CPU and GPU. You can get more info at <a href="https://www.fastcompression.com/download/Fastvideo_SDK_manual.pdf"   target="_blank">Fastvideo SDK Manual</a>.</p>

<h2>Image processing pipeline on GPU for machine vision applications</h2>
<ul>
  <li>Raw image capture (8-bit, 12-bit packed/unpacked, 16-bit)</li>
  <li>Import to GPU</li>
  <li>Raw data convert and unpack</li>
  <li>Linearization curve</li>
  <li>Bad Pixel Correction</li>  
  <li>Dark frame subtraction</li>  
  <li>Flat-Field Correction</li>  
  <li>White Balance</li>
  <li>Exposure Correction (brightness control)</li>  
  <li>Debayer with HQLI (5&times;5 window), DFPD (11&times;11), MG (23&times;23) algorithms</li>
  <li>Wavelet-based denoiser</li>  
  <li>Gamma</li>
  <li>JPEG / MJPEG encoding</li>
  <li>Output to monitor</li>  
  <li>Export from GPU to CPU memory</li>  
  <li>Storage of compressed data to SSD</li>    
</ul>

<h2>To build the project we need the following software for Windows</h2>
<ul>
  <li>Camera SDK (XIMEA, Basler, Baumer, JAI, Imperx, etc.)</li>
  <li>Fastvideo SDK (demo) ver.0.15.0.0</li>
  <li>NVIDIA CUDA-10.1</li>
  <li>Qt ver.5.13.1</li>
  <li>Compiler MSVC 2017</li>
</ul>

<h2>Minimum Hardware ans Software Requirements</h2>
<ul>
  <li>Windows7/10, 64-bit</li>
  <li>The latest NVIDIA driver</li>
  <li>NVIDIA GPU with Kepler architecture, 6xx series minimum</li>
  <li>NVIDIA GPU with 4 GB memory or better</li>
  <li>Intel Core i5 or better</li>
  <li>NVIDIA CUDA-10.1</li>
  <li>Compiler MSVC 2017 (MSVC 2015 is not compatible with CUDA-10.1)</li>
</ul>
<p>We also recommend to check PCI-Express bandwidth for Host-to-Device and Device-to-Host transfers. For GPU with Gen3 x16 it should be in the range of 10-12 GB/s. GPU memory size could be a bottleneck for high resolution cameras, so please check GPU memory usage in the software.</p>
<p>If you are working with images which reside on HDD, please place them on SSD or M2.</p>

<h2>Roadmap</h2>
<ul>
  <li>GPU pipeline for monochrome cameras - in progress</li>
  <li>H.264/H.265 encoders on GPU - in progress</li>  
  <li>Linux version - in progress</li>
  <li>Resize</li>
  <li>UnSharp Mask</li>
  <li>Rotation to an arbitrary angle</li>    
  <li>Support for JAI and Imperx cameras</li>
  <li>JPEG2000 encoder</li>
  <li>Realtime raw compression (lossless and/or lossy)</li>
  <li>Curves and Levels via 1D LUT</li>
  <li>Color correction with 3x3 matrix</li>  
  <li>Support of other color spaces</li>
  <li>3D LUT for HSV and RGB</li>
  <li>Defringe module</li>
  <li>DCP support</li>
  <li>LCP support (remap)</li>
  <li>Special version for NVIDIA Jetson hardware and L4T for CUDA-10.0</li>
  <li>Interoparability with external FFmpeg and GStreamer</li>
</ul>

<h2>Links</h2>
<ul>
  <li><a href="https://www.fastcompression.com/product/sdk.htm" target="_blank">Fastvideo SDK for Image & Video Processing</a></li>
</ul>

<h2>Fastvideo SDK Benchmarks</h2>
<ul>
  <li><a href="https://www.fastcompression.com/pub/2019/Fastvideo_SDK_benchmarks.pdf" target="_blank">Fastvideo SDK Benchmarks 2019</a></li>
  <li><a href="https://www.fastcompression.com/blog/jetson-benchmark-comparison.htm" target="_blank">Jetson Benchmark Comparison: Nano vs TX1 vs TX2 vs Xavier</a></li>
</ul>

<h2>Downloads</h2>
<ul>
  <li>Download <a href="https://www.fastcinemadng.com/download/download.html" target="_blank">Fast CinemaDNG Processor</a> software for Windows</li>
  <li>Download <a href="https://yadi.sk/d/7D3C3yx9wMLVNA">Fastvideo SDK (demo) for Windows-7/10, 64-bit</a> (valid till March 23, 2020)</li>
  <li>Download <a href="https://yadi.sk/d/7BADIJxdpBvz0A">Fastvideo SDK (demo) for Linux Ubuntu 18.04, 64-bit</a> (valid till March 23, 2020)</li>
  <li>Download <a href="https://yadi.sk/d/sM27Hl0JTbakhQ">Fastvideo SDK (demo) for NVIDIA Jetson Nano, TX2, Xavier</a> (valid till April 12, 2020)</li>
  <li>Download <a href="https://www.fastcompression.com/download/Fastvideo_SDK_manual.pdf" target="_blank">Fastvideo SDK Manual</a></li>
</ul>
