"""Check the installed Qt renderer and export a Windows icon from the SVG master."""
import io
import os
from pathlib import Path
import sys


def main() -> None:
    """Render without a display server; keep every standard Windows icon size."""
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PySide6 import QtCore, QtGui, QtSvg
    from PIL import Image

    app = QtGui.QGuiApplication([])
    renderer = QtSvg.QSvgRenderer(sys.argv[1])
    if not renderer.isValid():
        raise RuntimeError('The application SVG icon could not be loaded.')
    image = QtGui.QImage(256, 256, QtGui.QImage.Format_ARGB32)
    image.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(image)
    renderer.render(painter)
    painter.end()
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QIODevice.WriteOnly)
    if not image.save(buffer, 'PNG'):
        raise RuntimeError('Qt could not render the application icon.')
    Image.open(io.BytesIO(bytes(buffer.data()))).save(
        Path(sys.argv[2]), format='ICO', sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    print('Qt runtime and SVG icon are ready.')
    app.quit()


if __name__ == '__main__':
    main()
