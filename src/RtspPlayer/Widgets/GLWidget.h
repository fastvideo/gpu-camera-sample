#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include <QOpenGLFunctions>
#include <QGLShader>
#include <QGLShaderProgram>
#include <QMatrix4x4>
#include <QTimer>
#include <QElapsedTimer>

#include "common.h"

namespace Ui {
class GLWidget;
}

class GLWidget : public QGLWidget, private QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit GLWidget(QWidget *parent = nullptr);
    ~GLWidget();

    void setReceiver(AbstractReceiver *receiver);

    double fps() const;
    double bytesReaded() const;

	void start();
	void stop();

	QMap<QString, double> durations();

public slots:
    void onTimeout();

private:
    Ui::GLWidget *ui;
    AbstractReceiver *m_receiver = nullptr;
    QTimer m_timer;
    bool m_is_update = false;
    bool m_is_texupdate = false;
    PImage m_image;
    qint64 m_frameCount = 0;
    quint32 m_frameCount_Fps = 0;
    double m_fps = 0;
    QElapsedTimer m_timeFps;
    qint64 m_wait_timer_ms = 1500;
    quint64 m_LastBytesReaded = 0;
    double m_bytesReaded = 0;

	bool m_is_start = true;

	QMap<QString, double> m_durations;

    QMatrix4x4 m_modelview;
    QMatrix4x4 m_projection;
    QMatrix4x4 m_mvp;

    QGLShaderProgram m_shader_program;

    uint m_bindTex;
    uint m_bindTexU;
    uint m_bindTexV;
    int m_vecInt;
    int m_texInt;
    int m_mvpInt;
    int m_utexIntY;
    int m_utexIntU;
    int m_utexIntV;
    int m_rgbInt;

    std::vector< float > m_vertexBuffer;
    std::vector< float > m_textureBuffer;

    void drawGL();
    void generateTexture();
    void setViewport(float w, float h);

    // QWidget interface
protected:
    void paintEvent(QPaintEvent *event);

    // QGLWidget interface
protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL() override;
};

#endif // GLWIDGET_H
