#include <QGuiApplication>
#include <QApplication>
#include <QStyleFactory>

#include "MainWindow.h"

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    QApplication::setStyle(QStyleFactory::create(QStringLiteral("Fusion")));

    QPalette darkPalette;
    darkPalette.setColor(QPalette::Window, QColor(64,64,64));
    darkPalette.setColor(QPalette::WindowText, Qt::white);
    darkPalette.setColor(QPalette::Base, QColor(64,64,64));
    darkPalette.setColor(QPalette::AlternateBase, QColor(64,64,64));
    darkPalette.setColor(QPalette::ToolTipBase, Qt::white);
    darkPalette.setColor(QPalette::ToolTipText, Qt::white);
    darkPalette.setColor(QPalette::Text, Qt::white);
    darkPalette.setColor(QPalette::Button, QColor(64,64,64));
    darkPalette.setColor(QPalette::ButtonText, Qt::white);
    darkPalette.setColor(QPalette::BrightText, Qt::red);
    darkPalette.setColor(QPalette::Link, QColor(42, 130, 218));

    darkPalette.setColor(QPalette::Disabled, QPalette::Window, QColor(96,96,96));
    darkPalette.setColor(QPalette::Disabled, QPalette::WindowText, Qt::gray);
    darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, Qt::gray);
    darkPalette.setColor(QPalette::Disabled, QPalette::Base, QColor(96,96,96));
    darkPalette.setColor(QPalette::Disabled, QPalette::AlternateBase, QColor(96,96,96));
    darkPalette.setColor(QPalette::Disabled, QPalette::Button, QColor(96,96,96));

    darkPalette.setColor(QPalette::Highlight, QColor(130, 130, 130));
    darkPalette.setColor(QPalette::HighlightedText, Qt::black);

    QApplication::setPalette(darkPalette);

    darkPalette.setColor(QPalette::Window, QColor(0x929292)); //QColor("#929292")
    QApplication::setPalette(darkPalette, "QCheckBox");

    darkPalette.setColor(QPalette::Button, QColor(0x646464));//QColor("#646464")
    QApplication::setPalette(darkPalette, "QToolButton");

    darkPalette.setColor(QPalette::Window, QColor(0x929292));//QColor("#929292")
    darkPalette.setColor(QPalette::Base, QColor(80,80,80));
    QApplication::setPalette(darkPalette, "QLineEdit");

    darkPalette.setColor(QPalette::Mid, Qt::white);
    QApplication::setPalette(darkPalette, "QGroupBox");

    QStringList styleSheetList;
    styleSheetList << QStringLiteral("QToolTip { color: #ffffff; background-color: #929292; border: 1px solid white;}");
    styleSheetList << QStringLiteral("QDockWidget::title {text-align: left center; background: rgb(48, 48, 48);}");
    styleSheetList << QStringLiteral("QDockWidget {titlebar-close-icon: url(:/res/close.png); titlebar-normal-icon: url(:/res/undock.png);}");
    styleSheetList << QStringLiteral("QGroupBox {background-color: transparent; border: 1px solid gray; border-radius: 3px; margin-top: 7px;}");
    styleSheetList << QStringLiteral("QGroupBox::title {subcontrol-origin: margin; top: 0px; left: 15px;}");
    qApp->setStyleSheet(styleSheetList.join(QChar('\n')));

    MainWindow w;
    w.show();

    return app.exec();
}
