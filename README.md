# gpu-camera-sample
Camera sample application with realtime GPU processing

<h3>The software has the following architecture</h3>
<ul>
  <li>Thread for GUI and visualization (app main thread)</li>
  <li>Thread for image acquisition from a camera</li>
  <li>Thread to control CUDA-based image processing</li>
  <li>Thread for OpenGL rendering</li>
  <li>Thread for async data writing to SSD or streaming</li>
</ul>

<h3>Simple image processing pipeline on GPU for machine vision applications</h3>
<ul>
  <li>Raw image capture (8-bit, 12-bit packed/unpacked, 16-bit)</li>
  <li>Import to GPU</li>
  <li>Optional raw data convertion and unpacking</li>
  <li>Linearization curve</li>
  <li>Bad pixel removal</li>  
  <li>Dark frame subtraction</li>  
  <li>Flat-field correction</li>  
  <li>White Balance</li>
  <li>Exposure correction (brightness control)</li>  
  <li>Debayer with HQLI (5&times;5 window), DFPD (11&times;11), MG (23&times;23) algorithms</li>
  <li>Wavelet-based denoising</li>  
  <li>Gamma</li>
  <li>JPEG / MJPEG encoding</li>
  <li>Output to monitor</li>  
  <li>Export from GPU to CPU memory</li>  
  <li>Storage of compressed data to SSD</li>    
</ul>

<p>To accomplish the task we need the following</p>
<ul>
  <li>Camera SDK (XIMEA, Basler, Baumer, JAI, Imperx, etc.) for Windows</li>
  <li>Fastvideo SDK (demo) ver.0.15.0.2 for Windows</li>
  <li>NVIDIA CUDA-10.0 for Windows</li>
  <li>Qt ver.5.13.1 for Windows</li>
  <li>Compiler MSVC 2017</li>
</ul>

<p></p>
