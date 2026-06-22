"""Builders and parsers for SpikeGLX/CatGT/TPrime command-string fragments.

This module converts between human-readable spec dataclasses and the raw flag
strings consumed by CatGT and TPrime (for example ``-apfilter=...``,
``-xd=...``, and ``-bf=...``), and provides the Qt dialogs that let users edit
those fragments through tables and form fields. The build/parse helpers are
pure functions with no Qt dependency; the dialog classes wrap them for the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from .side_nav import SideNavStack
except ImportError:
    from neuropyguin.side_nav import SideNavStack


_STREAM_TO_JS = {
    "ni": 0,
    "obx": 1,
    "imec": 2,
}


@dataclass
class CatGTCommandSpec:
    """Editable view of a CatGT command fragment (folders, AP filter, gfix, extras)."""

    use_probe_folders: bool = True
    use_output_probe_folders: bool = True
    allow_missing_probes: bool = False
    allow_missing_trials: bool = False
    disable_auto_sync: bool = False
    use_ap_filter: bool = True
    ap_filter_type: str = "butter"
    ap_filter_order: int = 12
    ap_filter_highpass_hz: float = 300.0
    ap_filter_lowpass_hz: float = 10000.0
    use_gfix: bool = True
    gfix_amp_mv: float = 0.40
    gfix_slope_mv_per_sample: float = 0.10
    gfix_noise_mv: float = 0.02
    extra_flags: str = ""


@dataclass
class TPrimeExtractorSpec:
    """One CatGT event extractor (-xd/-xid/-xa/-xia) used by TPrime alignment."""

    mode: str = "xd"
    stream_kind: str = "ni"
    stream_index: int = 0
    word: int = 0
    value_a: float = 0.0
    value_b: float = 0.0
    debounce_ms: float = 0.0
    label: str = ""


@dataclass
class BitFieldExtractorSpec:
    """One CatGT bit-field extractor (-bf=js,ip,word,startbit,nbits,inarow)."""

    stream_kind: str = "ni"
    stream_index: int = 0
    word: int = 0
    start_bit: int = 0
    n_bits: int = 1
    inarow: int = 3


def _split_flags(raw: str) -> List[str]:
    """Split a flag string on runs of whitespace, dropping empty tokens."""
    return [part for part in re.split(r"\s+", str(raw).strip()) if part]


def _is_extractor_token(token: str) -> bool:
    """Return True if the token is a CatGT extractor flag (xd/xid/xa/xia/bf)."""
    clean = re.sub(r"\[.*?\]$", "", str(token).strip())
    return bool(re.fullmatch(r"-(xd|xid|xa|xia|bf)=(.+)", clean, flags=re.IGNORECASE))


def _is_bf_token(token: str) -> bool:
    """Return True if the token is a CatGT bit-field extractor flag (-bf=...)."""
    return bool(re.fullmatch(r"-bf=(.+)", str(token).strip(), flags=re.IGNORECASE))


def _stream_kind_from_js(js: int) -> str:
    """Map a CatGT stream selector (js: 0/1/2) back to its stream-kind name."""
    return {0: "ni", 1: "obx", 2: "imec"}.get(int(js), "ni")


def _fmt_number(value: float, decimals: int = 3) -> str:
    """Format a float with up to ``decimals`` places, trimming trailing zeros."""
    txt = f"{float(value):.{decimals}f}"
    txt = txt.rstrip("0").rstrip(".")
    if txt == "-0":
        txt = "0"
    return txt or "0"


def parse_channel_spec(raw: str) -> List[int]:
    """Parse a channel spec like ``"0-2,4"`` into a de-duplicated, ordered list."""
    values: List[int] = []
    for part in [chunk.strip() for chunk in str(raw).split(",") if chunk.strip()]:
        if "-" in part:
            left, _, right = part.partition("-")
            start = int(left.strip())
            stop = int(right.strip())
            if stop < start:
                start, stop = stop, start
            for value in range(start, stop + 1):
                if value not in values:
                    values.append(value)
        else:
            value = int(part)
            if value not in values:
                values.append(value)
    return values


def build_catgt_command_string(spec: CatGTCommandSpec) -> str:
    """Render a CatGTCommandSpec into a space-joined CatGT flag string."""
    parts: List[str] = []
    if spec.use_probe_folders:
        parts.append("-prb_fld")
    if spec.use_output_probe_folders:
        parts.append("-out_prb_fld")
    if spec.allow_missing_probes:
        parts.append("-prb_miss_ok")
    if spec.allow_missing_trials:
        parts.append("-t_miss_ok")
    if spec.disable_auto_sync:
        parts.append("-no_auto_sync")
    if spec.use_ap_filter:
        parts.append(
            "-apfilter="
            f"{spec.ap_filter_type},{int(spec.ap_filter_order)},"
            f"{_fmt_number(spec.ap_filter_highpass_hz)},"
            f"{_fmt_number(spec.ap_filter_lowpass_hz)}"
        )
    if spec.use_gfix:
        parts.append(
            "-gfix="
            f"{_fmt_number(spec.gfix_amp_mv, 3)},"
            f"{_fmt_number(spec.gfix_slope_mv_per_sample, 3)},"
            f"{_fmt_number(spec.gfix_noise_mv, 3)}"
        )
    if spec.extra_flags.strip():
        parts.extend(_split_flags(spec.extra_flags))
    return " ".join(parts)


def parse_catgt_command_string(raw: str) -> CatGTCommandSpec:
    """Parse a CatGT flag string into a CatGTCommandSpec, keeping unknown flags as extras."""
    spec = CatGTCommandSpec()
    extras: List[str] = []
    tokens = _split_flags(raw)
    for token in tokens:
        if token == "-prb_fld":
            spec.use_probe_folders = True
        elif token == "-out_prb_fld":
            spec.use_output_probe_folders = True
        elif token == "-prb_miss_ok":
            spec.allow_missing_probes = True
        elif token == "-t_miss_ok":
            spec.allow_missing_trials = True
        elif token == "-no_auto_sync":
            spec.disable_auto_sync = True
        elif token.startswith("-apfilter="):
            spec.use_ap_filter = True
            payload = token.split("=", 1)[1].split(",")
            if len(payload) >= 4:
                spec.ap_filter_type = payload[0] or spec.ap_filter_type
                spec.ap_filter_order = int(float(payload[1]))
                spec.ap_filter_highpass_hz = float(payload[2])
                spec.ap_filter_lowpass_hz = float(payload[3])
            else:
                extras.append(token)
        elif token.startswith("-gfix="):
            spec.use_gfix = True
            payload = token.split("=", 1)[1].split(",")
            if len(payload) >= 3:
                spec.gfix_amp_mv = float(payload[0])
                spec.gfix_slope_mv_per_sample = float(payload[1])
                spec.gfix_noise_mv = float(payload[2])
            else:
                extras.append(token)
        else:
            extras.append(token)

    if "-prb_fld" not in tokens:
        spec.use_probe_folders = False
    if "-out_prb_fld" not in tokens:
        spec.use_output_probe_folders = False
    if not any(token.startswith("-apfilter=") for token in tokens):
        spec.use_ap_filter = False
    if not any(token.startswith("-gfix=") for token in tokens):
        spec.use_gfix = False
    spec.extra_flags = " ".join(extras)
    return spec


def build_tostream_sync_params(stream_kind: str, stream_index: int) -> str:
    """Build a TPrime toStream sync-param token (``ni`` or ``imec0``/``obx0`` style)."""
    kind = str(stream_kind).strip().lower()
    if kind == "ni":
        return "ni"
    return f"{kind}{int(stream_index)}"


def parse_tostream_sync_params(raw: str) -> Tuple[str, int]:
    """Parse a TPrime toStream token into (stream_kind, stream_index)."""
    text = str(raw).strip().lower()
    if text == "ni":
        return "ni", 0
    match = re.fullmatch(r"(imec|obx)(\d+)", text)
    if match:
        return match.group(1), int(match.group(2))
    return "imec", 0


def build_tprime_extractor_string(specs: Sequence[TPrimeExtractorSpec], extra_flags: str = "") -> str:
    """Render extractor specs into CatGT -xd/-xid/-xa/-xia flags, appending raw extras."""
    parts: List[str] = []
    for spec in specs:
        mode = str(spec.mode).strip().lower()
        js = _STREAM_TO_JS.get(str(spec.stream_kind).strip().lower(), 0)
        stream_index = 0 if js == 0 else int(spec.stream_index)
        label_suffix = f"[{spec.label}]" if getattr(spec, "label", "") else ""
        if mode in {"xd", "xid"}:
            parts.append(
                f"-{mode}="
                f"{js},{stream_index},{int(spec.word)},{int(spec.value_a)},{_fmt_number(spec.debounce_ms)}"
                + label_suffix
            )
        else:
            parts.append(
                f"-{mode}="
                f"{js},{stream_index},{int(spec.word)},"
                f"{_fmt_number(spec.value_a)},"
                f"{_fmt_number(spec.value_b)},"
                f"{_fmt_number(spec.debounce_ms)}"
                + label_suffix
            )
    if str(extra_flags).strip():
        parts.extend(_split_flags(extra_flags))
    return " ".join(parts)


def parse_tprime_extractor_string(raw: str) -> Tuple[List[TPrimeExtractorSpec], str]:
    """Parse -xd/-xid/-xa/-xia flags into specs; return (specs, leftover-extras-string)."""
    specs: List[TPrimeExtractorSpec] = []
    extras: List[str] = []
    for token in _split_flags(raw):
        label = ""
        clean_token = token
        label_match = re.search(r"\[([^\]]*)\]$", token)
        if label_match:
            label = label_match.group(1)
            clean_token = token[: label_match.start()]
        match = re.fullmatch(r"-(xd|xid|xa|xia)=(.+)", clean_token, flags=re.IGNORECASE)
        if not match:
            extras.append(token)
            continue
        mode = match.group(1).lower()
        values = match.group(2).split(",")
        try:
            if mode in {"xd", "xid"} and len(values) >= 5:
                js = int(values[0])
                specs.append(
                    TPrimeExtractorSpec(
                        mode=mode,
                        stream_kind=_stream_kind_from_js(js),
                        stream_index=int(values[1]),
                        word=int(values[2]),
                        value_a=int(values[3]),
                        value_b=0.0,
                        debounce_ms=float(values[4]),
                        label=label,
                    )
                )
            elif mode in {"xa", "xia"} and len(values) >= 6:
                js = int(values[0])
                specs.append(
                    TPrimeExtractorSpec(
                        mode=mode,
                        stream_kind=_stream_kind_from_js(js),
                        stream_index=int(values[1]),
                        word=int(values[2]),
                        value_a=float(values[3]),
                        value_b=float(values[4]),
                        debounce_ms=float(values[5]),
                        label=label,
                    )
                )
            else:
                extras.append(token)
        except Exception:
                extras.append(token)
    return specs, " ".join(extras)


def strip_extractor_labels(raw: str) -> str:
    """Remove the trailing ``[label]`` annotations from an extractor string."""
    return re.sub(r"\[[^\]]*\]", "", raw)


def catgt_command_extractors(raw: str) -> str:
    """Return only the extractor flags (xd/xid/xa/xia/bf) from a CatGT command."""
    return " ".join([token for token in _split_flags(raw) if _is_extractor_token(token)])


def catgt_command_bf_extractors(raw: str) -> str:
    """Return only the bit-field (-bf) flags from a CatGT command."""
    return " ".join([token for token in _split_flags(raw) if _is_bf_token(token)])


def strip_catgt_extractors(raw: str) -> str:
    """Return the CatGT command with all extractor flags removed."""
    return " ".join([token for token in _split_flags(raw) if not _is_extractor_token(token)])


def strip_catgt_bf_extractors(raw: str) -> str:
    """Return the CatGT command with all bit-field (-bf) flags removed."""
    return " ".join([token for token in _split_flags(raw) if not _is_bf_token(token)])


def merge_extractors_into_catgt_command(raw_command: str, extractor_string: str) -> str:
    """Replace any existing extractor flags in the command with label-stripped ones."""
    base = strip_catgt_extractors(raw_command)
    clean_extractors = strip_extractor_labels(extractor_string.strip())
    parts = [part for part in [base.strip(), clean_extractors] if part]
    return " ".join(parts)


def build_bitfield_extractor_string(specs: Sequence[BitFieldExtractorSpec], extra_flags: str = "") -> str:
    """Render bit-field specs into CatGT -bf flags, appending raw extras."""
    parts: List[str] = []
    for spec in specs:
        js = _STREAM_TO_JS.get(str(spec.stream_kind).strip().lower(), 0)
        stream_index = 0 if js == 0 else int(spec.stream_index)
        parts.append(
            "-bf="
            f"{js},{stream_index},{int(spec.word)},{int(spec.start_bit)},{int(spec.n_bits)},{int(spec.inarow)}"
        )
    if str(extra_flags).strip():
        parts.extend(_split_flags(extra_flags))
    return " ".join(parts)


def parse_bitfield_extractor_string(raw: str) -> Tuple[List[BitFieldExtractorSpec], str]:
    """Parse -bf flags into specs; return (specs, leftover-extras-string)."""
    specs: List[BitFieldExtractorSpec] = []
    extras: List[str] = []
    for token in _split_flags(raw):
        match = re.fullmatch(r"-bf=(.+)", token, flags=re.IGNORECASE)
        if not match:
            extras.append(token)
            continue
        values = match.group(1).split(",")
        try:
            if len(values) >= 6:
                js = int(values[0])
                specs.append(
                    BitFieldExtractorSpec(
                        stream_kind=_stream_kind_from_js(js),
                        stream_index=int(values[1]),
                        word=int(values[2]),
                        start_bit=int(values[3]),
                        n_bits=int(values[4]),
                        inarow=int(values[5]),
                    )
                )
            else:
                extras.append(token)
        except Exception:
            extras.append(token)
    return specs, " ".join(extras)


def merge_bitfields_into_catgt_command(raw_command: str, bitfield_string: str) -> str:
    """Replace any existing -bf flags in the command with the given bit-field string."""
    base = strip_catgt_bf_extractors(raw_command)
    parts = [part for part in [base.strip(), bitfield_string.strip()] if part]
    return " ".join(parts)


class BitFieldBuilderDialog(QtWidgets.QDialog):
    """Dialog for building CatGT bit-field (-bf) extractor flags from a table."""

    def __init__(self, initial_bitfields: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Build CatGT bit-field (-bf) extractors")
        self.resize(860, 460)
        specs, extras = parse_bitfield_extractor_string(initial_bitfields)

        main = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Bit-field extractors decode contiguous digital bits into numeric values using CatGT "
            "(-bf=js,ip,word,startbit,nbits,inarow). These flags are written into the CatGT command only."
        )
        note.setWordWrap(True)
        main.addWidget(note)

        table_box = QtWidgets.QGroupBox("Bit-field extractors")
        table_layout = QtWidgets.QVBoxLayout(table_box)
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add bit-field")
        self.btn_remove = QtWidgets.QPushButton("Remove selected")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        btn_row.addStretch(1)
        table_layout.addLayout(btn_row)
        self.tbl = QtWidgets.QTableWidget(0, 6)
        self.tbl.setHorizontalHeaderLabels(["Stream", "Index", "Word", "Start bit", "Bit count", "In a row"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.tbl)
        help_label = QtWidgets.QLabel(
            "Use this when the signal is encoded as a binary number across adjacent bits in one digital word."
        )
        help_label.setWordWrap(True)
        table_layout.addWidget(help_label)
        main.addWidget(table_box, 1)

        self.ed_extra = QtWidgets.QLineEdit(extras)
        main.addWidget(QtWidgets.QLabel("Extra flags to append"))
        main.addWidget(self.ed_extra)

        self.preview = QtWidgets.QLineEdit()
        self.preview.setReadOnly(True)
        main.addWidget(QtWidgets.QLabel("Generated -bf flags"))
        main.addWidget(self.preview)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main.addWidget(buttons)

        self.btn_add.clicked.connect(self._add_row)
        self.btn_remove.clicked.connect(self._remove_selected_row)
        self.ed_extra.textChanged.connect(self._refresh_preview)

        if specs:
            for spec in specs:
                self._add_row(spec)
        else:
            self._add_row()
        self._refresh_preview()

    def _new_stream_combo(self, stream_kind: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItems(["ni", "obx", "imec"])
        combo.setCurrentText(stream_kind)
        combo.currentTextChanged.connect(self._refresh_preview)
        return combo

    def _new_spin(self, minimum: int, maximum: int, value: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(int(minimum), int(maximum))
        spin.setValue(int(value))
        spin.valueChanged.connect(self._refresh_preview)
        return spin

    def _row_of_widget(self, widget: QtWidgets.QWidget | None) -> int:
        if widget is None:
            return -1
        for row in range(self.tbl.rowCount()):
            for col in range(self.tbl.columnCount()):
                if self.tbl.cellWidget(row, col) is widget:
                    return row
        return -1

    def _sync_row_state_for_sender(self) -> None:
        sender = self.sender()
        if not isinstance(sender, QtWidgets.QWidget):
            return
        row = self._row_of_widget(sender)
        if row < 0:
            return
        stream_combo = self.tbl.cellWidget(row, 0)
        index_spin = self.tbl.cellWidget(row, 1)
        start_bit_spin = self.tbl.cellWidget(row, 3)
        bit_count_spin = self.tbl.cellWidget(row, 4)
        if isinstance(stream_combo, QtWidgets.QComboBox) and isinstance(index_spin, QtWidgets.QSpinBox):
            index_spin.setEnabled(stream_combo.currentText().strip().lower() != "ni")
        if isinstance(start_bit_spin, QtWidgets.QSpinBox) and isinstance(bit_count_spin, QtWidgets.QSpinBox):
            bit_count_spin.setMaximum(max(1, 16 - int(start_bit_spin.value())))
        self._refresh_preview()

    def _add_row(self, spec: BitFieldExtractorSpec | None = None) -> None:
        row = self.tbl.rowCount()
        self.tbl.insertRow(row)
        spec = spec or BitFieldExtractorSpec()
        stream_combo = self._new_stream_combo(spec.stream_kind)
        index_spin = self._new_spin(0, 31, spec.stream_index)
        word_spin = self._new_spin(0, 1024, spec.word)
        start_bit_spin = self._new_spin(0, 15, spec.start_bit)
        bit_count_spin = self._new_spin(1, 16, spec.n_bits)
        inarow_spin = self._new_spin(1, 1000, spec.inarow)
        self.tbl.setCellWidget(row, 0, stream_combo)
        self.tbl.setCellWidget(row, 1, index_spin)
        self.tbl.setCellWidget(row, 2, word_spin)
        self.tbl.setCellWidget(row, 3, start_bit_spin)
        self.tbl.setCellWidget(row, 4, bit_count_spin)
        self.tbl.setCellWidget(row, 5, inarow_spin)
        stream_combo.currentTextChanged.connect(self._sync_row_state_for_sender)
        start_bit_spin.valueChanged.connect(self._sync_row_state_for_sender)
        index_spin.setEnabled(spec.stream_kind != "ni")
        bit_count_spin.setMaximum(max(1, 16 - int(spec.start_bit)))
        self._refresh_preview()

    def _remove_selected_row(self) -> None:
        row = self.tbl.currentRow()
        if row < 0:
            row = self.tbl.rowCount() - 1
        if row >= 0:
            self.tbl.removeRow(row)
            self._refresh_preview()

    def _row_spec(self, row: int) -> BitFieldExtractorSpec | None:
        widgets = [self.tbl.cellWidget(row, col) for col in range(self.tbl.columnCount())]
        if not all(widgets):
            return None
        stream_combo, index_spin, word_spin, start_bit_spin, bit_count_spin, inarow_spin = widgets
        return BitFieldExtractorSpec(
            stream_kind=stream_combo.currentText().strip().lower(),
            stream_index=int(index_spin.value()),
            word=int(word_spin.value()),
            start_bit=int(start_bit_spin.value()),
            n_bits=int(bit_count_spin.value()),
            inarow=int(inarow_spin.value()),
        )

    def _refresh_preview(self) -> None:
        self.preview.setText(self.value())

    def value(self) -> str:
        """Return the generated -bf extractor string for the current table state."""
        specs: List[BitFieldExtractorSpec] = []
        for row in range(self.tbl.rowCount()):
            spec = self._row_spec(row)
            if spec is not None:
                specs.append(spec)
        return build_bitfield_extractor_string(specs, self.ed_extra.text().strip())


class CatGTStringBuilderDialog(QtWidgets.QDialog):
    """Dialog for building a CatGT command fragment from readable form fields."""

    def __init__(self, initial_command: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Build CatGT command string")
        self.resize(760, 440)
        spec = parse_catgt_command_string(initial_command)

        main = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Build the CatGT command fragment from readable fields. "
            "CAR mode and loccar radius stay controlled by the main preprocessing settings."
        )
        note.setWordWrap(True)
        main.addWidget(note)

        flags_box = QtWidgets.QGroupBox("Common flags")
        flags_layout = QtWidgets.QGridLayout(flags_box)
        self.ck_probe_folders = QtWidgets.QCheckBox("Use probe folders (-prb_fld)")
        self.ck_output_probe_folders = QtWidgets.QCheckBox("Create output probe folders (-out_prb_fld)")
        self.ck_missing_probes = QtWidgets.QCheckBox("Skip missing probes (-prb_miss_ok)")
        self.ck_missing_trials = QtWidgets.QCheckBox("Allow missing trials (-t_miss_ok)")
        self.ck_no_auto_sync = QtWidgets.QCheckBox("Disable auto sync extraction (-no_auto_sync)")
        self.ck_probe_folders.setChecked(spec.use_probe_folders)
        self.ck_output_probe_folders.setChecked(spec.use_output_probe_folders)
        self.ck_missing_probes.setChecked(spec.allow_missing_probes)
        self.ck_missing_trials.setChecked(spec.allow_missing_trials)
        self.ck_no_auto_sync.setChecked(spec.disable_auto_sync)
        flags_layout.addWidget(self.ck_probe_folders, 0, 0)
        flags_layout.addWidget(self.ck_output_probe_folders, 0, 1)
        flags_layout.addWidget(self.ck_missing_probes, 1, 0)
        flags_layout.addWidget(self.ck_missing_trials, 1, 1)
        flags_layout.addWidget(self.ck_no_auto_sync, 2, 0, 1, 2)
        main.addWidget(flags_box)

        options_row = QtWidgets.QHBoxLayout()

        filter_box = QtWidgets.QGroupBox("AP filter")
        filter_form = QtWidgets.QFormLayout(filter_box)
        self.ck_ap_filter = QtWidgets.QCheckBox("Enable AP filter")
        self.ck_ap_filter.setChecked(spec.use_ap_filter)
        self.cb_ap_type = QtWidgets.QComboBox()
        self.cb_ap_type.addItems(["butter", "biquad"])
        idx = self.cb_ap_type.findText(spec.ap_filter_type)
        if idx >= 0:
            self.cb_ap_type.setCurrentIndex(idx)
        self.sp_ap_order = QtWidgets.QSpinBox()
        self.sp_ap_order.setRange(1, 32)
        self.sp_ap_order.setValue(int(spec.ap_filter_order))
        self.sp_ap_high = QtWidgets.QDoubleSpinBox()
        self.sp_ap_high.setRange(0.0, 20000.0)
        self.sp_ap_high.setDecimals(2)
        self.sp_ap_high.setValue(float(spec.ap_filter_highpass_hz))
        self.sp_ap_low = QtWidgets.QDoubleSpinBox()
        self.sp_ap_low.setRange(0.0, 40000.0)
        self.sp_ap_low.setDecimals(2)
        self.sp_ap_low.setValue(float(spec.ap_filter_lowpass_hz))
        filter_form.addRow(self.ck_ap_filter)
        filter_form.addRow("Type", self.cb_ap_type)
        filter_form.addRow("Order", self.sp_ap_order)
        filter_form.addRow("High-pass corner (Hz)", self.sp_ap_high)
        filter_form.addRow("Low-pass corner (Hz)", self.sp_ap_low)
        options_row.addWidget(filter_box, 1)

        gfix_box = QtWidgets.QGroupBox("Artifact suppression")
        gfix_form = QtWidgets.QFormLayout(gfix_box)
        self.ck_gfix = QtWidgets.QCheckBox("Enable gfix")
        self.ck_gfix.setChecked(spec.use_gfix)
        self.sp_gfix_amp = QtWidgets.QDoubleSpinBox()
        self.sp_gfix_amp.setRange(0.0, 10.0)
        self.sp_gfix_amp.setDecimals(3)
        self.sp_gfix_amp.setValue(float(spec.gfix_amp_mv))
        self.sp_gfix_slope = QtWidgets.QDoubleSpinBox()
        self.sp_gfix_slope.setRange(0.0, 10.0)
        self.sp_gfix_slope.setDecimals(3)
        self.sp_gfix_slope.setValue(float(spec.gfix_slope_mv_per_sample))
        self.sp_gfix_noise = QtWidgets.QDoubleSpinBox()
        self.sp_gfix_noise.setRange(0.0, 10.0)
        self.sp_gfix_noise.setDecimals(3)
        self.sp_gfix_noise.setValue(float(spec.gfix_noise_mv))
        gfix_form.addRow(self.ck_gfix)
        gfix_form.addRow("Amplitude (mV)", self.sp_gfix_amp)
        gfix_form.addRow("Slope (mV / sample)", self.sp_gfix_slope)
        gfix_form.addRow("Noise (mV)", self.sp_gfix_noise)
        options_row.addWidget(gfix_box, 1)
        main.addLayout(options_row)

        self.ed_extra = QtWidgets.QLineEdit(spec.extra_flags)
        main.addWidget(QtWidgets.QLabel("Extra CatGT flags"))
        main.addWidget(self.ed_extra)

        self.preview = QtWidgets.QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(84)
        main.addWidget(QtWidgets.QLabel("Generated command fragment"))
        main.addWidget(self.preview)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main.addWidget(buttons)

        for widget in [
            self.ck_probe_folders,
            self.ck_output_probe_folders,
            self.ck_missing_probes,
            self.ck_missing_trials,
            self.ck_no_auto_sync,
            self.ck_ap_filter,
            self.cb_ap_type,
            self.sp_ap_order,
            self.sp_ap_high,
            self.sp_ap_low,
            self.ck_gfix,
            self.sp_gfix_amp,
            self.sp_gfix_slope,
            self.sp_gfix_noise,
        ]:
            if hasattr(widget, "stateChanged"):
                widget.stateChanged.connect(self._refresh_preview)
            if hasattr(widget, "currentTextChanged"):
                widget.currentTextChanged.connect(self._refresh_preview)
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._refresh_preview)
        self.ed_extra.textChanged.connect(self._refresh_preview)
        self.ck_ap_filter.stateChanged.connect(self._sync_enabled_state)
        self.ck_gfix.stateChanged.connect(self._sync_enabled_state)
        self._sync_enabled_state()
        self._refresh_preview()

    def _sync_enabled_state(self) -> None:
        ap_enabled = self.ck_ap_filter.isChecked()
        for widget in [self.cb_ap_type, self.sp_ap_order, self.sp_ap_high, self.sp_ap_low]:
            widget.setEnabled(ap_enabled)
        gfix_enabled = self.ck_gfix.isChecked()
        for widget in [self.sp_gfix_amp, self.sp_gfix_slope, self.sp_gfix_noise]:
            widget.setEnabled(gfix_enabled)

    def _refresh_preview(self) -> None:
        self.preview.setPlainText(build_catgt_command_string(self.spec()))

    def spec(self) -> CatGTCommandSpec:
        """Collect the current widget values into a CatGTCommandSpec."""
        return CatGTCommandSpec(
            use_probe_folders=self.ck_probe_folders.isChecked(),
            use_output_probe_folders=self.ck_output_probe_folders.isChecked(),
            allow_missing_probes=self.ck_missing_probes.isChecked(),
            allow_missing_trials=self.ck_missing_trials.isChecked(),
            disable_auto_sync=self.ck_no_auto_sync.isChecked(),
            use_ap_filter=self.ck_ap_filter.isChecked(),
            ap_filter_type=self.cb_ap_type.currentText().strip(),
            ap_filter_order=int(self.sp_ap_order.value()),
            ap_filter_highpass_hz=float(self.sp_ap_high.value()),
            ap_filter_lowpass_hz=float(self.sp_ap_low.value()),
            use_gfix=self.ck_gfix.isChecked(),
            gfix_amp_mv=float(self.sp_gfix_amp.value()),
            gfix_slope_mv_per_sample=float(self.sp_gfix_slope.value()),
            gfix_noise_mv=float(self.sp_gfix_noise.value()),
            extra_flags=self.ed_extra.text().strip(),
        )

    def value(self) -> str:
        """Return the generated CatGT command fragment for the current form state."""
        return build_catgt_command_string(self.spec())


class TPrimeStringBuilderDialog(QtWidgets.QDialog):
    """Dialog for building the TPrime reference stream and event-extractor strings."""

    def __init__(
        self,
        initial_to_stream: str,
        initial_extractors: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("compactDialog", True)
        self.setWindowTitle("Build TPrime stream and extractor strings")
        self.resize(1100, 760)
        stream_kind, stream_index = parse_tostream_sync_params(initial_to_stream)
        specs, extras = parse_tprime_extractor_string(initial_extractors)
        fixed_font = QtGui.QFont("Consolas", 9)

        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(18, 16, 18, 16)
        main.setSpacing(14)
        note = QtWidgets.QLabel(
            "Choose the TPrime reference stream and define CatGT event extractors without writing raw "
            "-xd/-xid/-xa/-xia strings. Use rising and falling rows together when you want TPrime-aligned "
            "pulse onsets and offsets to recover full TTL durations."
        )
        note.setObjectName("SectionHint")
        note.setWordWrap(True)
        main.addWidget(note)

        self.section_nav = SideNavStack(
            "Sections",
            "Only the selected panel is shown so the extractor table can use the available space.",
        )
        main.addWidget(self.section_nav, 1)

        def _new_page() -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
            page = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(page)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(12)
            return page, layout

        ref_page, ref_page_layout = _new_page()
        ref_box = QtWidgets.QGroupBox("Reference stream")
        ref_box.setProperty("settingsSection", True)
        ref_layout = QtWidgets.QVBoxLayout(ref_box)
        ref_layout.setSpacing(10)
        ref_hint = QtWidgets.QLabel(
            "Pick the stream TPrime will align to using TPrime names such as ni, imec0, or obx0. "
            "In most Neuropixels workflows this is an imec stream."
        )
        ref_hint.setObjectName("SectionHint")
        ref_hint.setWordWrap(True)
        ref_layout.addWidget(ref_hint)
        ref_form = QtWidgets.QFormLayout()
        ref_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldsStayAtSizeHint)
        ref_form.setFormAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        ref_form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        ref_form.setHorizontalSpacing(12)
        ref_form.setVerticalSpacing(10)
        self.cb_stream_kind = QtWidgets.QComboBox()
        self.cb_stream_kind.addItems(["imec", "ni", "obx"])
        self.cb_stream_kind.setCurrentText(stream_kind)
        self.cb_stream_kind.setMaximumWidth(140)
        self.sp_stream_index = QtWidgets.QSpinBox()
        self.sp_stream_index.setRange(0, 31)
        self.sp_stream_index.setValue(int(stream_index))
        self.sp_stream_index.setMaximumWidth(90)
        ref_form.addRow("Type", self.cb_stream_kind)
        ref_form.addRow("Index", self.sp_stream_index)
        ref_layout.addLayout(ref_form)
        ref_layout.addStretch(1)
        ref_page_layout.addWidget(ref_box)
        ref_page_layout.addStretch(1)
        self.section_nav.add_page("Reference stream", ref_page)

        preset_page, preset_page_layout = _new_page()
        preset_box = QtWidgets.QGroupBox("Common setup: NI analog events to imec")
        preset_box.setProperty("settingsSection", True)
        preset_layout = QtWidgets.QVBoxLayout(preset_box)
        preset_layout.setSpacing(10)
        preset_hint = QtWidgets.QLabel(
            "Fast path for the common case where NI analog event channels should be aligned onto an imec timeline. "
            "Enter NI XA channel numbers, thresholds, and whether you also want falling edges. Applying the preset "
            "sets `toStream_sync_params` to the chosen imec stream and replaces existing NI analog rows."
        )
        preset_hint.setObjectName("SectionHint")
        preset_hint.setWordWrap(True)
        preset_layout.addWidget(preset_hint)
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.setSpacing(18)
        preset_left = QtWidgets.QFormLayout()
        preset_left.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldsStayAtSizeHint)
        preset_left.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        preset_left.setHorizontalSpacing(12)
        preset_left.setVerticalSpacing(10)
        preset_right = QtWidgets.QFormLayout()
        preset_right.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldsStayAtSizeHint)
        preset_right.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        preset_right.setHorizontalSpacing(12)
        preset_right.setVerticalSpacing(10)
        self.sp_preset_imec_index = QtWidgets.QSpinBox()
        self.sp_preset_imec_index.setRange(0, 31)
        self.sp_preset_imec_index.setValue(int(stream_index) if stream_kind == "imec" else 0)
        self.sp_preset_imec_index.setMaximumWidth(90)
        self.ed_preset_channels = QtWidgets.QLineEdit("0-2")
        self.ed_preset_channels.setPlaceholderText("Example: 0-2,4")
        self.ed_preset_channels.setMaximumWidth(180)
        self.sp_preset_th1 = QtWidgets.QDoubleSpinBox()
        self.sp_preset_th1.setRange(-10.0, 10.0)
        self.sp_preset_th1.setDecimals(3)
        self.sp_preset_th1.setValue(1.0)
        self.sp_preset_th1.setMaximumWidth(110)
        self.sp_preset_th2 = QtWidgets.QDoubleSpinBox()
        self.sp_preset_th2.setRange(-10.0, 10.0)
        self.sp_preset_th2.setDecimals(3)
        self.sp_preset_th2.setValue(0.0)
        self.sp_preset_th2.setMaximumWidth(110)
        self.sp_preset_pulse_ms = QtWidgets.QDoubleSpinBox()
        self.sp_preset_pulse_ms.setRange(0.0, 100000.0)
        self.sp_preset_pulse_ms.setDecimals(3)
        self.sp_preset_pulse_ms.setValue(0.0)
        self.sp_preset_pulse_ms.setMaximumWidth(110)
        self.ck_preset_include_falling = QtWidgets.QCheckBox("Include falling edges (xia)")
        self.ck_preset_include_falling.setChecked(True)
        self.btn_apply_ni_analog_preset = QtWidgets.QPushButton("Apply NI analog preset")
        self.btn_apply_ni_analog_preset.setProperty("role", "primary")
        preset_left.addRow("toStream imec index", self.sp_preset_imec_index)
        preset_left.addRow("Threshold 1 (V)", self.sp_preset_th1)
        preset_left.addRow("Pulse ms", self.sp_preset_pulse_ms)
        preset_right.addRow("NI XA channels", self.ed_preset_channels)
        preset_right.addRow("Threshold 2 (V)", self.sp_preset_th2)
        preset_right.addRow("", self.ck_preset_include_falling)
        preset_row.addLayout(preset_left)
        preset_row.addLayout(preset_right)
        preset_row.addStretch(1)
        preset_layout.addLayout(preset_row)
        preset_btn_row = QtWidgets.QHBoxLayout()
        preset_btn_row.addStretch(1)
        preset_btn_row.addWidget(self.btn_apply_ni_analog_preset)
        preset_layout.addLayout(preset_btn_row)
        preset_page_layout.addWidget(preset_box)
        preset_page_layout.addStretch(1)
        self.section_nav.add_page("NI analog preset", preset_page)

        extract_page, extract_page_layout = _new_page()
        table_box = QtWidgets.QGroupBox("Event extractors")
        table_box.setProperty("heroCard", True)
        table_layout = QtWidgets.QVBoxLayout(table_box)
        table_layout.setSpacing(10)
        table_hint = QtWidgets.QLabel(
            "Each row defines one CatGT extractor. Digital rows use xd/xid for rising/falling edges on bits. "
            "Analog rows use xa/xia for rising/falling threshold crossings. Use imec rows for digital events only."
        )
        table_hint.setObjectName("SectionHint")
        table_hint.setWordWrap(True)
        table_layout.addWidget(table_hint)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        add_label = QtWidgets.QLabel("Quick add")
        add_label.setObjectName("FieldTitle")
        self.btn_add_digital = QtWidgets.QPushButton("Digital rise")
        self.btn_add_digital_fall = QtWidgets.QPushButton("Digital fall")
        self.btn_add_analog = QtWidgets.QPushButton("Analog rise")
        self.btn_add_analog_fall = QtWidgets.QPushButton("Analog fall")
        self.btn_remove = QtWidgets.QPushButton("Remove selected")
        self.btn_add_digital.setProperty("role", "secondary")
        self.btn_add_digital_fall.setProperty("role", "secondary")
        self.btn_add_analog.setProperty("role", "secondary")
        self.btn_add_analog_fall.setProperty("role", "secondary")
        self.btn_remove.setProperty("role", "ghost")
        btn_row.addWidget(add_label)
        btn_row.addWidget(self.btn_add_digital)
        btn_row.addWidget(self.btn_add_digital_fall)
        btn_row.addWidget(self.btn_add_analog)
        btn_row.addWidget(self.btn_add_analog_fall)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_remove)
        table_layout.addLayout(btn_row)
        self.tbl = QtWidgets.QTableWidget(0, 8)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setShowGrid(False)
        self.tbl.setHorizontalHeaderLabels(
            ["Mode", "Stream", "Index", "Word", "Bit / Th1", "Th2", "Pulse ms", "Label"]
        )
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.verticalHeader().setDefaultSectionSize(34)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl.setMinimumHeight(430)
        self.tbl.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        header = self.tbl.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(7, QtWidgets.QHeaderView.Stretch)
        table_layout.addWidget(self.tbl)
        help_label = QtWidgets.QLabel(
            "Digital modes: 'Bit / Th1' is the bit number and 'Th2' is ignored. "
            "Analog modes: 'Bit / Th1' and 'Th2' are thresholds in volts. "
            "Set Pulse ms to 0 to report all detected edges. "
            "To capture full TTL widths, add both a rising and a falling extractor on the same line."
        )
        help_label.setObjectName("SectionHint")
        help_label.setWordWrap(True)
        table_layout.addWidget(help_label)
        extract_page_layout.addWidget(table_box, 1)
        self.section_nav.add_page("Event extractors", extract_page)

        extra_page, extra_page_layout = _new_page()
        extra_box = QtWidgets.QGroupBox("Extra extractor flags")
        extra_box.setProperty("settingsSection", True)
        extra_layout = QtWidgets.QVBoxLayout(extra_box)
        extra_layout.setSpacing(8)
        extra_hint = QtWidgets.QLabel(
            "Optional raw CatGT extractor flags to append unchanged. Leave empty if you only use the table above."
        )
        extra_hint.setObjectName("SectionHint")
        extra_hint.setWordWrap(True)
        extra_layout.addWidget(extra_hint)
        self.ed_extra = QtWidgets.QLineEdit(extras)
        self.ed_extra.setPlaceholderText("Example: -bf=0,0,8,3,4,3")
        extra_layout.addWidget(self.ed_extra)
        extra_page_layout.addWidget(extra_box)
        extra_page_layout.addStretch(1)
        self.section_nav.add_page("Extra flags", extra_page)

        preview_page, preview_page_layout = _new_page()
        preview_box = QtWidgets.QGroupBox("Generated values")
        preview_box.setProperty("settingsSection", True)
        preview_layout = QtWidgets.QVBoxLayout(preview_box)
        preview_layout.setSpacing(10)
        preview_hint = QtWidgets.QLabel(
            "These are the exact values that will be written back into the preprocessing form."
        )
        preview_hint.setObjectName("SectionHint")
        preview_hint.setWordWrap(True)
        preview_layout.addWidget(preview_hint)

        to_stream_row = QtWidgets.QHBoxLayout()
        to_stream_label = QtWidgets.QLabel("Reference stream (`toStream_sync_params`)")
        to_stream_label.setObjectName("FieldTitle")
        self.btn_copy_to_stream = QtWidgets.QPushButton("Copy")
        self.btn_copy_to_stream.setProperty("role", "ghost")
        to_stream_row.addWidget(to_stream_label)
        to_stream_row.addStretch(1)
        to_stream_row.addWidget(self.btn_copy_to_stream)
        preview_layout.addLayout(to_stream_row)
        self.ed_to_stream_preview = QtWidgets.QPlainTextEdit()
        self.ed_to_stream_preview.setReadOnly(True)
        self.ed_to_stream_preview.setFixedHeight(58)
        self.ed_to_stream_preview.setFont(fixed_font)
        preview_layout.addWidget(self.ed_to_stream_preview)

        extract_row = QtWidgets.QHBoxLayout()
        extract_label = QtWidgets.QLabel("Extractor string (legacy setting name: `tPrime_ni_ex_list`)")
        extract_label.setObjectName("FieldTitle")
        self.btn_copy_extract = QtWidgets.QPushButton("Copy")
        self.btn_copy_extract.setProperty("role", "ghost")
        extract_row.addWidget(extract_label)
        extract_row.addStretch(1)
        extract_row.addWidget(self.btn_copy_extract)
        preview_layout.addLayout(extract_row)
        self.ed_extract_preview = QtWidgets.QPlainTextEdit()
        self.ed_extract_preview.setReadOnly(True)
        self.ed_extract_preview.setMinimumHeight(180)
        self.ed_extract_preview.setFont(fixed_font)
        preview_layout.addWidget(self.ed_extract_preview)
        preview_page_layout.addWidget(preview_box, 1)
        self.section_nav.add_page("Generated values", preview_page)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main.addWidget(buttons)

        self.btn_add_digital.clicked.connect(lambda: self._add_row("xd"))
        self.btn_add_digital_fall.clicked.connect(lambda: self._add_row("xid"))
        self.btn_add_analog.clicked.connect(lambda: self._add_row("xa"))
        self.btn_add_analog_fall.clicked.connect(lambda: self._add_row("xia"))
        self.btn_remove.clicked.connect(self._remove_selected_row)
        self.btn_copy_to_stream.clicked.connect(lambda: self._copy_preview(self.ed_to_stream_preview.toPlainText()))
        self.btn_copy_extract.clicked.connect(lambda: self._copy_preview(self.ed_extract_preview.toPlainText()))
        self.btn_apply_ni_analog_preset.clicked.connect(self._apply_ni_analog_preset)
        self.cb_stream_kind.currentTextChanged.connect(self._sync_stream_controls)
        self.cb_stream_kind.currentTextChanged.connect(self._refresh_preview)
        self.sp_stream_index.valueChanged.connect(self._refresh_preview)
        self.ed_extra.textChanged.connect(self._refresh_preview)
        self._default_placeholder_active = not specs

        if specs:
            for spec in specs:
                self._add_row(spec.mode, spec, clear_placeholder=False)
        else:
            self._add_row("xd", clear_placeholder=False)
        self._sync_stream_controls()
        self._refresh_preview()
        self.section_nav.setCurrentIndex(2)

    def _new_mode_combo(self, mode: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItems(["xd", "xid", "xa", "xia"])
        combo.setCurrentText(mode)
        combo.currentTextChanged.connect(self._refresh_preview)
        return combo

    def _new_stream_combo(self, stream_kind: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItems(["ni", "obx", "imec"])
        combo.setCurrentText(stream_kind)
        combo.currentTextChanged.connect(self._refresh_preview)
        return combo

    def _new_index_spin(self, value: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(0, 31)
        spin.setValue(int(value))
        spin.valueChanged.connect(self._refresh_preview)
        return spin

    def _new_word_spin(self, value: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(-1, 1024)
        spin.setValue(int(value))
        spin.valueChanged.connect(self._refresh_preview)
        return spin

    def _new_value_spin(self, value: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-10000.0, 10000.0)
        spin.setDecimals(3)
        spin.setValue(float(value))
        spin.valueChanged.connect(self._refresh_preview)
        return spin

    def _configure_row_widgets(
        self,
        mode_combo: QtWidgets.QComboBox,
        stream_combo: QtWidgets.QComboBox,
        index_spin: QtWidgets.QSpinBox,
        value_a: QtWidgets.QDoubleSpinBox,
        value_b: QtWidgets.QDoubleSpinBox,
    ) -> None:
        mode = mode_combo.currentText().strip().lower()
        stream_kind = stream_combo.currentText().strip().lower()
        index_spin.setEnabled(stream_kind != "ni")
        if mode in {"xd", "xid"}:
            value_a.setDecimals(0)
            value_a.setRange(0, 31)
            value_b.setEnabled(False)
        else:
            value_a.setDecimals(3)
            value_a.setRange(-10.0, 10.0)
            value_b.setEnabled(True)
            value_b.setDecimals(3)
            value_b.setRange(-10.0, 10.0)

    def _row_of_widget(self, widget: QtWidgets.QWidget | None) -> int:
        if widget is None:
            return -1
        for row in range(self.tbl.rowCount()):
            for col in range(self.tbl.columnCount()):
                if self.tbl.cellWidget(row, col) is widget:
                    return row
        return -1

    def _sync_row_state(self, row: int) -> None:
        mode_combo = self.tbl.cellWidget(row, 0)
        stream_combo = self.tbl.cellWidget(row, 1)
        index_spin = self.tbl.cellWidget(row, 2)
        value_a = self.tbl.cellWidget(row, 4)
        value_b = self.tbl.cellWidget(row, 5)
        if not isinstance(mode_combo, QtWidgets.QComboBox):
            return
        if not isinstance(stream_combo, QtWidgets.QComboBox):
            return
        if not isinstance(index_spin, QtWidgets.QSpinBox):
            return
        if not isinstance(value_a, QtWidgets.QDoubleSpinBox):
            return
        if not isinstance(value_b, QtWidgets.QDoubleSpinBox):
            return
        self._configure_row_widgets(mode_combo, stream_combo, index_spin, value_a, value_b)
        self._refresh_preview()

    def _sync_row_state_for_sender(self) -> None:
        sender = self.sender()
        if not isinstance(sender, QtWidgets.QWidget):
            return
        row = self._row_of_widget(sender)
        if row >= 0:
            self._sync_row_state(row)

    def _add_row(self, mode: str, spec: TPrimeExtractorSpec | None = None, *, clear_placeholder: bool = True) -> None:
        if clear_placeholder and getattr(self, "_default_placeholder_active", False) and self.tbl.rowCount() == 1:
            self.tbl.removeRow(0)
            self._default_placeholder_active = False
        row = self.tbl.rowCount()
        self.tbl.insertRow(row)
        spec = spec or TPrimeExtractorSpec(mode=mode)
        mode_combo = self._new_mode_combo(spec.mode)
        stream_combo = self._new_stream_combo(spec.stream_kind)
        index_spin = self._new_index_spin(spec.stream_index)
        word_spin = self._new_word_spin(spec.word)
        value_a = self._new_value_spin(spec.value_a)
        value_b = self._new_value_spin(spec.value_b)
        duration_spin = self._new_value_spin(spec.debounce_ms)
        label_edit = QtWidgets.QLineEdit(getattr(spec, "label", "") or "")
        label_edit.setPlaceholderText("e.g. laser, reward")
        label_edit.textChanged.connect(self._refresh_preview)
        self.tbl.setCellWidget(row, 0, mode_combo)
        self.tbl.setCellWidget(row, 1, stream_combo)
        self.tbl.setCellWidget(row, 2, index_spin)
        self.tbl.setCellWidget(row, 3, word_spin)
        self.tbl.setCellWidget(row, 4, value_a)
        self.tbl.setCellWidget(row, 5, value_b)
        self.tbl.setCellWidget(row, 6, duration_spin)
        self.tbl.setCellWidget(row, 7, label_edit)
        mode_combo.currentTextChanged.connect(self._sync_row_state_for_sender)
        stream_combo.currentTextChanged.connect(self._sync_row_state_for_sender)
        self._sync_row_state(row)

    def _remove_selected_row(self) -> None:
        row = self.tbl.currentRow()
        if row < 0:
            row = self.tbl.rowCount() - 1
        if row >= 0:
            self.tbl.removeRow(row)
            if self.tbl.rowCount() == 0:
                self._default_placeholder_active = False
            self._refresh_preview()

    def _remove_rows_matching(self, predicate) -> None:
        for row in range(self.tbl.rowCount() - 1, -1, -1):
            spec = self._row_spec(row)
            if spec is not None and predicate(spec):
                self.tbl.removeRow(row)

    def _copy_preview(self, text: str) -> None:
        QtWidgets.QApplication.clipboard().setText(text.strip())

    def _sync_stream_controls(self) -> None:
        self.sp_stream_index.setEnabled(self.cb_stream_kind.currentText().strip().lower() != "ni")

    def _apply_ni_analog_preset(self) -> None:
        try:
            channels = parse_channel_spec(self.ed_preset_channels.text().strip())
        except Exception:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid channels",
                "Enter NI XA channels as comma-separated numbers or ranges, for example `0-2,4`.",
            )
            return
        if not channels:
            QtWidgets.QMessageBox.warning(
                self,
                "No channels",
                "Enter at least one NI XA channel number.",
            )
            return

        self.cb_stream_kind.setCurrentText("imec")
        self.sp_stream_index.setValue(int(self.sp_preset_imec_index.value()))

        self._remove_rows_matching(
            lambda spec: spec.stream_kind == "ni" and spec.mode in {"xa", "xia"}
        )
        if getattr(self, "_default_placeholder_active", False) and self.tbl.rowCount() == 1:
            self.tbl.removeRow(0)
        self._default_placeholder_active = False

        th1 = float(self.sp_preset_th1.value())
        th2 = float(self.sp_preset_th2.value())
        pulse_ms = float(self.sp_preset_pulse_ms.value())
        include_falling = self.ck_preset_include_falling.isChecked()

        for channel in channels:
            base_spec = TPrimeExtractorSpec(
                mode="xa",
                stream_kind="ni",
                stream_index=0,
                word=int(channel),
                value_a=th1,
                value_b=th2,
                debounce_ms=pulse_ms,
            )
            self._add_row("xa", base_spec)
            if include_falling:
                falling_spec = TPrimeExtractorSpec(
                    mode="xia",
                    stream_kind="ni",
                    stream_index=0,
                    word=int(channel),
                    value_a=th1,
                    value_b=th2,
                    debounce_ms=pulse_ms,
                )
                self._add_row("xia", falling_spec)
        self._refresh_preview()
        self.section_nav.setCurrentIndex(2)

    def _row_spec(self, row: int) -> TPrimeExtractorSpec | None:
        mode_combo = self.tbl.cellWidget(row, 0)
        stream_combo = self.tbl.cellWidget(row, 1)
        index_spin = self.tbl.cellWidget(row, 2)
        word_spin = self.tbl.cellWidget(row, 3)
        value_a = self.tbl.cellWidget(row, 4)
        value_b = self.tbl.cellWidget(row, 5)
        duration_spin = self.tbl.cellWidget(row, 6)
        label_edit = self.tbl.cellWidget(row, 7)
        widgets = [mode_combo, stream_combo, index_spin, word_spin, value_a, value_b, duration_spin]
        if not all(widgets):
            return None
        label = label_edit.text().strip() if isinstance(label_edit, QtWidgets.QLineEdit) else ""
        return TPrimeExtractorSpec(
            mode=mode_combo.currentText().strip().lower(),
            stream_kind=stream_combo.currentText().strip().lower(),
            stream_index=int(index_spin.value()),
            word=int(word_spin.value()),
            value_a=float(value_a.value()),
            value_b=float(value_b.value()),
            debounce_ms=float(duration_spin.value()),
            label=label,
        )

    def _refresh_preview(self) -> None:
        to_stream, ex_string = self.values()
        self.ed_to_stream_preview.setPlainText(to_stream)
        self.ed_extract_preview.setPlainText(ex_string)

    def values(self) -> Tuple[str, str]:
        """Return (toStream_sync_params, extractor_string) for the current dialog state."""
        specs: List[TPrimeExtractorSpec] = []
        for row in range(self.tbl.rowCount()):
            spec = self._row_spec(row)
            if spec is not None:
                specs.append(spec)
        to_stream = build_tostream_sync_params(self.cb_stream_kind.currentText(), int(self.sp_stream_index.value()))
        extractors = build_tprime_extractor_string(specs, self.ed_extra.text().strip())
        return to_stream, extractors
