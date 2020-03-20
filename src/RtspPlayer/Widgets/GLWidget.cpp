#include "GLWidget.h"
#include "ui_glwidget.h"

#include <stdio.h>

#include <QPainter>
#include <QGuiApplication>
#include <QScreen>

#include <fastvideo_sdk.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

///////////////////////////

void RGB2Yuv420p(unsigned char *yuv,
							   unsigned char *rgb,
							   int width,
							   int height)
{
  const size_t image_size = width * height;
  unsigned char *dst_y = yuv;
  unsigned char *dst_u = yuv + image_size;
  unsigned char *dst_v = yuv + image_size * 5 / 4;

	// Y plane
	for(size_t i = 0; i < image_size; i++)
	{
		int r = rgb[3 * i];
		int g = rgb[3 * i + 1];
		int b = rgb[3 * i + 2];
		*dst_y++ = ((67316 * r + 132154 * g + 25666 * b) >> 18 ) + 16;
	}

	// U and V plane
	for(size_t y = 0; y < height; y+=2)
	{
		for(size_t x = 0; x < width; x+=2)
		{
			const size_t i = y * width + x;
			int r = rgb[3 * i];
			int g = rgb[3 * i + 1];
			int b = rgb[3 * i + 2];
			*dst_u++ = ((-38856 * r - 76282 * g + 115138 * b ) >> 18 ) + 128;
			*dst_v++ = ((115138 * r - 96414 * g - 18724 * b) >> 18 ) + 128;
		}
	}
}

///////////////////////////

void addPt(std::vector< float >& buf, float x1, float x2, float x3)
{
    buf.push_back(x1);
    buf.push_back(x2);
    buf.push_back(x3);
}

void addPt(std::vector< float >& buf, float x1, float x2)
{
    buf.push_back(x1);
    buf.push_back(x2);
}

///////////////////////////

GLWidget::GLWidget(QWidget *parent) :
    QGLWidget(parent),
    QOpenGLFunctions(),
    ui(new Ui::GLWidget)
{
    ui->setupUi(this);

	setAutoFillBackground(false);

    connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));
	m_timer.start(2);
    m_timeFps.start();
}

GLWidget::~GLWidget()
{
	if(m_cudaRgb){
		cudaFree(m_cudaRgb);
	}

    delete ui;
}

void GLWidget::setReceiver(AbstractReceiver *receiver)
{
    m_receiver = receiver;
}

double GLWidget::fps() const
{
    return m_fps;
}

double GLWidget::bytesReaded() const
{
	return m_bytesReaded;
}

void GLWidget::start()
{
	m_is_start = true;
}

void GLWidget::stop()
{
	m_is_start = false;
}

QMap<QString, double> GLWidget::durations()
{
	return m_durations;
}

void GLWidget::onTimeout()
{
    if(m_receiver){
        if(m_receiver->isFrameExists()){

			if(m_is_start){

				if(m_timeFps.elapsed() > m_wait_timer_ms){
					double cnt = m_frameCount_Fps;
					double time = m_timeFps.elapsed();
					double bytesReaded = m_receiver->bytesReaded() - m_LastBytesReaded;
					m_LastBytesReaded = m_receiver->bytesReaded();
					m_frameCount_Fps = 0;
					m_timeFps.restart();

					if(time > 0){
						m_fps = cnt / time * 1000.;
						m_bytesReaded = bytesReaded / time * 1000.;
					}else{
						m_fps = 0;
						m_bytesReaded = 0;
					}
				}

				m_image = m_receiver->takeFrame();
				m_is_update = true;
				m_is_texupdate = true;
				m_frameCount++;
				m_frameCount_Fps++;

			}else{
				m_receiver->takeFrame();
			}

            //qDebug("frames %d, w=%d; h=%d               \r", m_frameCount, m_image->width, m_image->height);
        }
    }

    if(m_is_update){
        m_is_update = false;
        generateTexture();
        update();
	}
}

inline void qmat2float(const QMatrix4x4& mat, float* data, int len = 16)
{
    for(int i = 0; i < len; ++i)
        data[i] = static_cast<float>(mat.constData()[i]);
}

void GLWidget::setViewport(float w, float h)
{
    float ar = 1.f * w / h;
    m_projection.setToIdentity();
    m_projection.ortho(-ar, ar, -1, 1, 1, 10);
}

void GLWidget::drawGL()
{
    setViewport(width(), height());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0, 0, 0, 1);

    m_shader_program.bind();

    float mvp[16];

    m_modelview.setToIdentity();
    m_modelview.translate(0, 0, -2);

    float arim = 1;
    float ar = (float)width() / height();

    if(!m_image.get()){
        return;
    }

    if(!m_image->empty()){
     arim = (float)m_image->width/m_image->height;
    }

    if(ar > arim){
		m_modelview.scale(arim, 1, 1);
    }else{
		m_modelview.scale(ar, ar / arim, 1);
    }

    m_mvp = m_projection * m_modelview;

    qmat2float(m_mvp, mvp);

    glUniformMatrix4fv(m_mvpInt, 1, false, mvp);

	GLfloat iw = m_image->width;
	GLfloat ih = m_image->height;

	auto starttime = getNow();

	//glActiveTexture(GL_TEXTURE1);
	glEnable(GL_TEXTURE_2D);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, iw, ih, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);

    glEnableVertexAttribArray(m_vecInt);
    glEnableVertexAttribArray(m_texInt);
    glVertexAttribPointer(m_texInt, 2, GL_FLOAT, false, 0, m_textureBuffer.data());
    glVertexAttribPointer(m_vecInt, 3, GL_FLOAT, false, 0, m_vertexBuffer.data());
    glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLsizei)m_vertexBuffer.size() / 3);
    glDisableVertexAttribArray(m_vecInt);
    glDisableVertexAttribArray(m_texInt);

    glDisable(GL_TEXTURE_2D);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glFinish();

	m_durations["output_duration"] = getDuration(starttime);
}

bool GLWidget::initCudaBuffer()
{
	if(m_cudaRgb && m_image->width == m_prevWidth && m_image->height == m_prevHeight){
		return true;
	}
	releaseCudaBuffer();

	size_t sz = m_image->width * m_image->height * 3;

	cudaError_t err = cudaMalloc(&m_cudaRgb, sz);

	return err == cudaSuccess;
}

void GLWidget::releaseCudaBuffer()
{
	if(m_cudaRgb){
		cudaFree(m_cudaRgb);
	}
	m_cudaRgb = nullptr;
}

void GLWidget::generateTexture()
{
    if(!m_image.get())
        return;

    if(!m_is_texupdate)
        return;
    m_is_texupdate = false;

	unsigned char *data = NULL;
	size_t pboBufferSize = 0;
	cudaError_t error = cudaSuccess;

	auto starttime = getNow();

	int width = m_image->width;
	int height = m_image->height;

	if(!initCudaBuffer()){
		return;
	}

	void *img = m_cudaRgb;

	if(m_image->type == Image::YUV || m_image->type == Image::NV12){
		if(!m_sdiConverter.convertToRgb(m_image, m_cudaRgb)){
			return;
		}
	}else if(m_image->type == Image::RGB){
		error = cudaMemcpy(m_cudaRgb, m_image->rgb.data(), m_image->rgb.size(), cudaMemcpyHostToDevice);
		if(error != cudaSuccess){
			return;
		}
	}else if(m_image->type == Image::GRAY){
		// not yet supported
		return;
	}

	GLint bsize;
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glBufferData(GL_PIXEL_UNPACK_BUFFER, 3 * sizeof(unsigned char) * width * height, NULL, GL_STREAM_COPY);
	glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize);
	struct cudaGraphicsResource* cuda_pbo_resource = 0;

	if(img == nullptr)
		return;

	error = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_buffer, cudaGraphicsMapFlagsWriteDiscard);
	if(error != cudaSuccess)
	{
		qDebug("Cannot register CUDA Graphic Resource: %s\n", cudaGetErrorString(error));
		return;
	}

	if((error = cudaGraphicsMapResources( 1, &cuda_pbo_resource, 0 ) ) != cudaSuccess)
	{
		qDebug("cudaGraphicsMapResources failed: %s\n", cudaGetErrorString(error) );
		return;
	}

	if((error = cudaGraphicsResourceGetMappedPointer( (void **)&data, &pboBufferSize, cuda_pbo_resource ) ) != cudaSuccess )
	{
		qDebug("cudaGraphicsResourceGetMappedPointer failed: %s\n", cudaGetErrorString(error) );
		return;
	}

	if(pboBufferSize < ( width * height * 3 * sizeof(unsigned char) ))
	{
		qDebug("cudaGraphicsResourceGetMappedPointer failed: %s\n", cudaGetErrorString(error) );
		return;
	}

	if((error = cudaMemcpy( data, img, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToDevice ) ) != cudaSuccess)
	{
		qDebug("cudaMemcpy failed: %s\n", cudaGetErrorString(error) );
		return;
	}

	if((error = cudaGraphicsUnmapResources( 1, &cuda_pbo_resource, 0 ) ) != cudaSuccess )
	{
		 qDebug("cudaGraphicsUnmapResources failed: %s\n", cudaGetErrorString(error) );
		 return;
	}

	if(cuda_pbo_resource)
	{
		if((error  = cudaGraphicsUnregisterResource(cuda_pbo_resource))!= cudaSuccess)
		{
			qDebug("Cannot unregister CUDA Graphic Resource: %s\n", cudaGetErrorString(error));
			return;
		}
		cuda_pbo_resource = 0;
	}

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	m_prevWidth = m_image->width;
	m_prevHeight = m_image->height;
	m_prevType = m_image->type;

	m_durations["generate_texture_rgb"] = getDuration(starttime);
}

void GLWidget::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event)
    makeCurrent();
    drawGL();
    doneCurrent();

    QPainter painter(this);

    painter.end();
}

void GLWidget::initializeGL()
{
    QGLWidget::initializeGL();
    QOpenGLFunctions::initializeOpenGLFunctions();

	bool res = false;
//    glGenTextures(1, &m_bindTex);
//    glGenTextures(1, &m_bindTexU);
//    glGenTextures(1, &m_bindTexV);

	res = m_shader_program.addShaderFromSourceCode(QGLShader::Vertex,
                                            "attribute vec3 aVec;\n"
                                            "attribute vec2 aTex;\n"
                                            "uniform mat4 uMvp;\n"
                                            "varying vec2 vTex;\n"
                                            "void main(){\n"
                                            "    gl_Position = uMvp * vec4(aVec, 1);\n"
                                            "    vTex = aTex;\n"
                                            "}");

	res = m_shader_program.addShaderFromSourceCode(QGLShader::Fragment,
                                             "//precision highp float;\n"
                                            "varying vec2 vTex;                                                         \n"
                                            "uniform sampler2D uTexY;                                                   \n"
                                            "void main(){                                                               \n"
											"   gl_FragColor = texture2D(uTexY, vTex);                              \n"
                                            "}");

    m_shader_program.link();
    m_shader_program.bind();

    m_vecInt = m_shader_program.attributeLocation("aVec");
    m_texInt = m_shader_program.attributeLocation("aTex");
    m_mvpInt = m_shader_program.uniformLocation("uMvp");
    m_utexIntY = m_shader_program.uniformLocation("uTexY");
//    m_utexIntU = m_shader_program.uniformLocation("uTexU");
//    m_utexIntV = m_shader_program.uniformLocation("uTexV");
//    m_rgbInt = m_shader_program.uniformLocation("rgb");

    addPt(m_vertexBuffer, -1, -1, 0);
    addPt(m_vertexBuffer, -1, 1, 0);
    addPt(m_vertexBuffer, 1, -1, 0);
    addPt(m_vertexBuffer, 1, 1, 0);

    addPt(m_textureBuffer, 0, 1);
    addPt(m_textureBuffer, 0, 0);
    addPt(m_textureBuffer, 1, 1);
    addPt(m_textureBuffer, 1, 0);

	QSize sz = QGuiApplication::primaryScreen()->size() * 2;

	GLint bsize;
	glGenTextures(1, &texture);
	glGenBuffers(1, &pbo_buffer);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, GLsizeiptr(3 * sizeof(unsigned char)) * sz.width() * sz.height(), nullptr, GL_STREAM_COPY);
	glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

//	QImage im;
//	im.load("test.jpeg");
//	if(!im.isNull()){
//		im = im.convertToFormat(QImage::Format_RGB888);
//		m_image.reset(new Image(im.width(), im.height(), Image::YUV));
//		RGB2Yuv420p(m_image->yuv.data(), im.bits(), im.width(), im.height());
////		size_t sz = im.width() * im.height() * 3;
////		memcpy(m_image->rgb.data(), im.bits(), sz);
//		m_is_texupdate = true;
//		m_is_update = true;
//	}

}

void GLWidget::resizeGL(int w, int h)
{
    QGLWidget::resizeGL(w, h);

    glViewport(0, 0, w, h);
    setViewport(w, h);
}

void GLWidget::paintGL()
{
    QGLWidget::paintGL();

//    drawGL();
}
