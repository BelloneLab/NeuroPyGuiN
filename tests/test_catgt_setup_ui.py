"""Regression checks for CatGT dialog sizing, saved values, and scalable branding."""

from pathlib import Path
import os
import xml.etree.ElementTree as ET

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

from neuropyguin.string_builders import CatGTSetupDialog
from neuropyguin.styles import build_app_palette, build_app_qss


@pytest.fixture(scope="module")
def application():
    """Keep one QApplication alive for the geometry and SVG renderer checks."""
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def dialog(application):
    """Use non-default LF settings to detect silent resets when reopening setup."""
    widget = CatGTSetupDialog(
        initial_command="-prb_fld -out_prb_fld -apfilter=butter,12,300,10000",
        initial_to_stream="imec0", initial_extractors="-xd=0,0,8,0,500",
        initial_output_streams="both", initial_gate="current", initial_trigger="current",
        initial_probe="current", initial_probe_count=2, initial_probe_ids="0-1",
        initial_lf_lowpass_hz=200.0, initial_lf_downsample=15,
        initial_car_mode="gbldmx", initial_loccar_min_um=40.0, initial_loccar_max_um=160.0,
    )
    yield widget
    widget.close()
    widget.deleteLater()
    application.processEvents()


@pytest.mark.parametrize("theme", ["Light", "Dark"])
@pytest.mark.parametrize("size", [(720, 580), (800, 700), (1120, 880)])
def test_processing_page_fits_window_and_keeps_preview_readable(application, dialog, theme, size):
    """A small display must scroll settings instead of forcing the dialog offscreen."""
    application.setStyleSheet(build_app_qss(theme))
    application.setPalette(build_app_palette(theme))
    dialog.tabs.setCurrentIndex(1)
    dialog.resize(*size)
    dialog.show()
    for _ in range(4):
        application.processEvents()
    assert dialog.width() <= size[0]
    assert dialog.height() <= size[1]
    assert dialog.tabs.currentWidget().horizontalScrollBar().maximum() == 0
    assert dialog.command_preview.viewport().height() >= 70
    assert dialog.command_preview.horizontalScrollBar().maximum() == 0
    assert dialog.rect().contains(dialog.btn_copy_command.mapTo(dialog, QtCore.QPoint(0, 0)))


def test_saved_filters_and_contextual_controls(application, dialog):
    """Friendly reference labels and conditional fields must preserve pipeline values."""
    dialog.show()
    dialog.tabs.setCurrentIndex(1)
    application.processEvents()
    assert dialog.values()["catgt_lf_lowpass_hz"] == 200.0
    assert dialog.values()["catgt_lf_downsample"] == 15
    assert not dialog.local_reference_field.isVisible()
    dialog.cb_car_mode.setCurrentIndex(dialog.cb_car_mode.findData("loccar"))
    assert dialog.local_reference_field.isVisible()
    assert dialog.values()["catgt_car_mode"] == "loccar"
    assert "-loccar_um=40,160" in dialog.command_preview.toPlainText()
    dialog.cb_output_streams.setCurrentIndex(dialog.cb_output_streams.findData("ap"))
    assert not dialog.catgt_panel.sp_lfp_lowpass.isEnabled()
    dialog.cb_output_streams.setCurrentIndex(dialog.cb_output_streams.findData("both"))
    assert dialog.catgt_panel.sp_lfp_lowpass.isEnabled()
    assert "-lffilter=butter,12,0,200" in dialog.command_preview.toPlainText()
    dialog.tabs.setCurrentIndex(0)
    application.processEvents()
    assert not dialog.gate_range_row.isVisible()
    dialog.cb_gate_mode.setCurrentIndex(dialog.cb_gate_mode.findData("range"))
    assert dialog.gate_range_row.isVisible()
    dialog.sp_gate_first.setValue(2)
    dialog.sp_gate_last.setValue(4)
    assert dialog.values()["gate_string"] == "2,4"
    dialog.btn_copy_command.click()
    assert application.clipboard().text() == dialog.command_preview.toPlainText()


@pytest.mark.parametrize("asset", ["neuropyguin-icon.svg", "neuropyguin-logo.svg"])
def test_branding_is_self_contained_vector_and_renders(application, asset):
    """Neither raster images nor externally resolved fonts belong in the SVG masters."""
    path = Path(__file__).resolve().parents[1] / "neuropyguin" / "assets" / asset
    root = ET.parse(path).getroot()
    assert not any(node.tag.rsplit("}", 1)[-1] in {"image", "text"} for node in root.iter())
    renderer = QtSvg.QSvgRenderer(str(path))
    assert renderer.isValid()
    for size in (24, 512, 2048):
        image = QtGui.QImage(size, size, QtGui.QImage.Format_ARGB32_Premultiplied)
        image.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(image)
        renderer.render(painter)
        painter.end()
        assert image.pixelColor(size // 2, size // 2).alpha() > 0
        if "icon" in asset:
            assert image.pixelColor(0, 0).alpha() == 0
