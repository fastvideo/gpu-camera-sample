#include "GLWidget.h"
#include "ui_glwidget.h"

#include <stdio.h>

#include <QPainter>

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

    if(m_image->type == Image::RGB){
        glUniform1i(m_rgbInt, 1);
	}else if(m_image->type == Image::GRAY){
		glUniform1i(m_rgbInt, 2);
	}else{
        glUniform1i(m_rgbInt, 0);
    }

	auto starttime = getNow();

    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_bindTex);
    glUniform1i(m_utexIntY, 0);

    if(m_image->type == Image::YUV){
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, m_bindTexU);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, m_bindTexV);
        glUniform1i(m_utexIntU, 1);
        glUniform1i(m_utexIntV, 2);
    }

    glEnableVertexAttribArray(m_vecInt);
    glEnableVertexAttribArray(m_texInt);
    glVertexAttribPointer(m_texInt, 2, GL_FLOAT, false, 0, m_textureBuffer.data());
    glVertexAttribPointer(m_vecInt, 3, GL_FLOAT, false, 0, m_vertexBuffer.data());
    glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLsizei)m_vertexBuffer.size() / 3);
    glDisableVertexAttribArray(m_vecInt);
    glDisableVertexAttribArray(m_texInt);

    glDisable(GL_TEXTURE_2D);

	m_durations["output_duration"] = getDuration(starttime);
}

void GLWidget::generateTexture()
{
    if(!m_image.get())
        return;

    if(!m_is_texupdate)
        return;
    m_is_texupdate = false;

	bool newTex = false;

	if(m_prevWidth != m_image->width || m_prevHeight != m_image->height || m_prevType != m_image->type){
		newTex = true;
	}

	auto starttime = getNow();

	m_prevWidth = m_image->width;
	m_prevHeight = m_image->height;
	m_prevType = m_image->type;

    if(m_image->type == Image::YUV){
		glBindTexture(GL_TEXTURE_2D, m_bindTex);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		if(newTex){
			glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, m_image->width, m_image->height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, m_image->Y.data());
		}else {
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_image->width, m_image->height, GL_LUMINANCE, GL_UNSIGNED_BYTE, m_image->Y.data());
		}

        glBindTexture(GL_TEXTURE_2D, m_bindTexU);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		if(newTex){
			glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, m_image->width/2, m_image->height/2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, m_image->U.data());
		}else {
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_image->width/2, m_image->height/2, GL_LUMINANCE, GL_UNSIGNED_BYTE, m_image->U.data());
		}

        glBindTexture(GL_TEXTURE_2D, m_bindTexV);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		if(newTex){
			glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, m_image->width/2, m_image->height/2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, m_image->V.data());
		}else {
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_image->width/2, m_image->height/2, GL_LUMINANCE, GL_UNSIGNED_BYTE, m_image->V.data());
		}

		m_durations["generate_texture_yuv"] = getDuration(starttime);
	}else if(m_image->type == Image::RGB){
		glBindTexture(GL_TEXTURE_2D, m_bindTex);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		if(newTex){
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_image->width, m_image->height, 0, GL_RGB, GL_UNSIGNED_BYTE, m_image->rgb.data());
		}else{
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_image->width, m_image->height, GL_RGB, GL_UNSIGNED_BYTE, m_image->rgb.data());
		}

		m_durations["generate_texture_rgb"] = getDuration(starttime);
	}else if(m_image->type == Image::GRAY){
		glBindTexture(GL_TEXTURE_2D, m_bindTex);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		if(newTex){
			glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, m_image->width, m_image->height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, m_image->rgb.data());
		}else{
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_image->width, m_image->height, GL_LUMINANCE, GL_UNSIGNED_BYTE, m_image->rgb.data());
		}

		m_durations["generate_texture_gray"] = getDuration(starttime);
	}
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

    glGenTextures(1, &m_bindTex);
    glGenTextures(1, &m_bindTexU);
    glGenTextures(1, &m_bindTexV);

    m_shader_program.addShaderFromSourceCode(QGLShader::Vertex,
                                            "attribute vec3 aVec;\n"
                                            "attribute vec2 aTex;\n"
                                            "uniform mat4 uMvp;\n"
                                            "varying vec2 vTex;\n"
                                            "void main(){\n"
                                            "    gl_Position = uMvp * vec4(aVec, 1);\n"
                                            "    vTex = aTex;\n"
                                            "}");

    m_shader_program.addShaderFromSourceCode(QGLShader::Fragment,
                                             "//precision highp float;\n"
                                            "varying vec2 vTex;                                                         \n"
                                            "uniform sampler2D uTexY;                                                   \n"
                                            "uniform sampler2D uTexU;                                                   \n"
                                            "uniform sampler2D uTexV;                                                   \n"
                                            "uniform int rgb;                                                           \n"
                                            "vec3 getRgb(vec3 yuv)                                                      \n"
                                            " {                                                                         \n"
                                            "     vec3 vec;                                                             \n"
                                            "                                                                           \n"
                                            "     vec.x = yuv.x + 1.402 * (yuv.z - 0.5);                                \n"
                                            "     vec.y = yuv.x - 0.344 * (yuv.y - 0.5) - 0.714 * (yuv.z - 0.5);        \n"
                                            "     vec.z = yuv.x + 1.772 * (yuv.y - 0.5);                                \n"
                                            "     return vec;                                                           \n"
                                            " }                                                                         \n"
                                            "void main(){                                                               \n"
											"   if(rgb  == 1){                                                          \n"
                                            "       gl_FragColor = texture2D(uTexY, vTex);                              \n"
											"   }																		\n"
											"   if(rgb == 2){															\n"
											"		gl_FragColor = texture2D(uTexY, vTex).bbba;							\n"
											"	}																		\n"
											"   if(rgb == 0){                                                           \n"
											"       float y = texture2D(uTexY, vTex).b;                                 \n"
											"       float u = texture2D(uTexU, vTex).b;                                 \n"
											"       float v = texture2D(uTexV, vTex).b;                                 \n"
											"       gl_FragColor = vec4(getRgb(vec3(y, u, v)), 1);                      \n"
                                            "   }                                                                       \n"
                                            "}");

    m_shader_program.link();
    m_shader_program.bind();

    m_vecInt = m_shader_program.attributeLocation("aVec");
    m_texInt = m_shader_program.attributeLocation("aTex");
    m_mvpInt = m_shader_program.uniformLocation("uMvp");
    m_utexIntY = m_shader_program.uniformLocation("uTexY");
    m_utexIntU = m_shader_program.uniformLocation("uTexU");
    m_utexIntV = m_shader_program.uniformLocation("uTexV");
    m_rgbInt = m_shader_program.uniformLocation("rgb");

    addPt(m_vertexBuffer, -1, -1, 0);
    addPt(m_vertexBuffer, -1, 1, 0);
    addPt(m_vertexBuffer, 1, -1, 0);
    addPt(m_vertexBuffer, 1, 1, 0);

    addPt(m_textureBuffer, 0, 1);
    addPt(m_textureBuffer, 0, 0);
    addPt(m_textureBuffer, 1, 1);
    addPt(m_textureBuffer, 1, 0);
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
