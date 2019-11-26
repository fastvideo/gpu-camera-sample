# gpu-camera-sample
Camera sample application with realtime GPU processing

<p>This is an implementation of standard image processing pipeline for camera applications. Processing is done on NVIDIA GPU to speedup the performance. The software could also work with raw images in PGM format and you can utilize these images for testing or if you don't have a camera.</p>

<p>Currently the software is working with XIMEA cameras. Soon we are going to add support of JAI and Imperx cameras.</p>

<h3>The software has the following architecture</h3>
<ul>
  <li>Thread for GUI and visualization (app main thread)</li>
  <li>Thread for image acquisition from a camera</li>
  <li>Thread to control CUDA-based image processing</li>
  <li>Thread for OpenGL rendering</li>
  <li>Thread for async data writing to SSD or streaming</li>
</ul>

<h3>Image processing pipeline on GPU for machine vision applications</h3>
<ul>
  <li>Raw image capture (8-bit, 12-bit packed/unpacked, 16-bit)</li>
  <li>Import to GPU</li>
  <li>Optional raw data convertion and unpacking</li>
  <li>Linearization curve</li>
  <li>Bad Pixel Correction</li>  
  <li>Dark frame subtraction</li>  
  <li>Flat-Field Correction</li>  
  <li>White Balance</li>
  <li>Exposure Correction (brightness control)</li>  
  <li>Debayer with HQLI (5&times;5 window), DFPD (11&times;11), MG (23&times;23) algorithms</li>
  <li>Wavelet-based denoising</li>  
  <li>Gamma</li>
  <li>JPEG / MJPEG encoding</li>
  <li>Output to monitor</li>  
  <li>Export from GPU to CPU memory</li>  
  <li>Storage of compressed data to SSD</li>    
</ul>

<h3>To accomplish the task we need the following</h3>
<ul>
  <li>Camera SDK (XIMEA, Basler, Baumer, JAI, Imperx, etc.) for Windows</li>
  <li>Fastvideo SDK (demo) ver.0.15.0.2 for Windows</li>
  <li>NVIDIA CUDA-10.0 for Windows</li>
  <li>Qt ver.5.13.1 for Windows</li>
  <li>Compiler MSVC 2017</li>
</ul>

<h2>Roadmap</h2>
<ul>
  <li>GPU pipeline for monochrome cameras - in progress</li>
  <li>H.264/H.265 encoders - in progress</li>  
  <li>Linux version - in progress</li>
  <li>Resize</li>
  <li>UnSharp Mask</li>
  <li>Rotation to an arbitrary angle</li>    
  <li>Support for JAI and Imperx cameras</li>
  <li>JPEG2000 encoder</li>
  <li>Realtime raw compression (lossless and/or lossy)</li>
  <li>Curves and Levels via 1D LUT</li>
  <li>Color correction with 3x3 matrix</li>  
  <li>Other color spaces</li>
  <li>3D LUT for HSV and RGB</li>
  <li>Defringe</li>
  <li>Special version for NVIDIA Jetson hardware</li>
  <li>Interoparability with external FFmpeg and GStreamer</li>
</ul>

<p></p>
