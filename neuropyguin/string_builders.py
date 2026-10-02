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
    use_lfp_filter: bool = False
    lfp_lowpass_hz: float = 300.0
    lfp_downsample: int = 12
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
    if spec.use_lfp_filter:
        parts.append(f"-lffilter=butter,12,0,{_fmt_number(spec.lfp_lowpass_hz)}")
        parts.append(f"-ap2lf_dwnsmp={int(spec.lfp_downsample)}")
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
        elif token.startswith("-lffilter="):
            payload = token.split("=", 1)[1].split(",")
            if len(payload) >= 4:
                spec.use_lfp_filter = True
                spec.lfp_lowpass_hz = float(payload[3])
            else:
                extras.append(token)
        elif token.startswith("-ap2lf_dwnsmp="):
            try:
                spec.lfp_downsample = int(float(token.split("=", 1)[1]))
                spec.use_lfp_filter = True
            except ValueError:
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


def build_catgt_command_preview(
    *,
    run_name: str,
    input_directory: str,
    output_directory: str,
    gate_string: str,
    trigger_string: str,
    probe_string: str,
    output_streams: str,
    car_mode: str,
    loccar_min_um: float,
    loccar_max_um: float,
    command_flags: str,
    extractor_flags: str,
) -> str:
    """Render the full CatGT argument line shown by the combined builder."""
    streams = list({"ap": ["-ap"], "lfp": ["-lf"], "both": ["-ap", "-lf"]}.get(
        str(output_streams).strip().lower(), ["-ap"]
    ))
    event_tokens = _split_flags(extractor_flags)
    if any(
        re.match(r"-(?:xd|xid|xa|xia|bf)=0,", re.sub(r"\[[^\]]*\]$", "", token), re.IGNORECASE)
        for token in event_tokens
    ) and "-ni" not in streams:
        streams.append("-ni")
    car = str(car_mode).strip().lower()
    if car == "loccar":
        car_flag = f"-loccar_um={_fmt_number(loccar_min_um)},{_fmt_number(loccar_max_um)}"
    elif car in {"gbldmx", "gblcar"}:
        car_flag = f"-{car}"
    else:
        car_flag = ""
    identity = [
        f"-dir={input_directory or '<input folder>'}",
        f"-run={run_name or '<run name>'}",
        f"-g={gate_string}",
        f"-t={trigger_string}",
        f"-prb={probe_string}",
        *streams,
    ]
    parts = [*identity, car_flag, str(command_flags).strip(), strip_extractor_labels(str(extractor_flags))]
    parts = [part for part in parts if part]
    parts.append(f"-dest={output_directory or '<output folder>'}")
    return "runit.sh '" + " ".join(parts) + "'"


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


class CatGTStringBuilderPanel(QtWidgets.QWidget):
    """Reusable CatGT filtering and flag editor for standalone or combined dialogs."""

    commandChanged = QtCore.Signal(str)

    def __init__(self, initial_command: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        spec = parse_catgt_command_string(initial_command)

        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(16)

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
        self.filter_cards = QtWidgets.QWidget()
        self.filter_grid = QtWidgets.QGridLayout(self.filter_cards)
        self.filter_grid.setContentsMargins(0, 0, 0, 0)
        self.filter_grid.setSpacing(16)

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
        lfp_box = QtWidgets.QGroupBox("LFP filter")
        lfp_form = QtWidgets.QFormLayout(lfp_box)
        self.ck_lfp_filter = QtWidgets.QCheckBox("Create or filter LFP output")
        self.ck_lfp_filter.setChecked(spec.use_lfp_filter)
        self.sp_lfp_lowpass = QtWidgets.QDoubleSpinBox()
        self.sp_lfp_lowpass.setRange(1.0, 1000.0)
        self.sp_lfp_lowpass.setDecimals(1)
        self.sp_lfp_lowpass.setValue(float(spec.lfp_lowpass_hz))
        self.cb_lfp_downsample = QtWidgets.QComboBox()
        self.cb_lfp_downsample.addItems(["2", "3", "4", "5", "6", "10", "12", "15", "20", "25", "30"])
        self.cb_lfp_downsample.setCurrentText(str(spec.lfp_downsample))
        lfp_form.addRow(self.ck_lfp_filter)
        lfp_form.addRow("Low-pass corner (Hz)", self.sp_lfp_lowpass)
        lfp_form.addRow("Downsample factor", self.cb_lfp_downsample)
        filter_form.addRow(self.ck_ap_filter)
        filter_form.addRow("Type", self.cb_ap_type)
        filter_form.addRow("Order", self.sp_ap_order)
        filter_form.addRow("High-pass corner (Hz)", self.sp_ap_high)
        filter_form.addRow("Low-pass corner (Hz)", self.sp_ap_low)

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
        self._cards = [filter_box, lfp_box, gfix_box]
        self._card_columns = 0
        for form in [filter_form, lfp_form, gfix_form]:
            # Labels above editors keep units readable at both laptop and desktop widths.
            form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            form.setContentsMargins(16, 20, 16, 16)
            form.setVerticalSpacing(8)
            form.setFormAlignment(QtCore.Qt.AlignTop)
        for card in self._cards:
            card.setMinimumWidth(220)
            card.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self._reflow_filter_cards(1000)
        main.addWidget(self.filter_cards)

        self.btn_more_options = QtWidgets.QToolButton()
        self.btn_more_options.setText("Folder layout and extra flags")
        self.btn_more_options.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btn_more_options.setArrowType(QtCore.Qt.RightArrow)
        self.btn_more_options.setCheckable(True)
        main.addWidget(self.btn_more_options)
        self.more_options = QtWidgets.QWidget()
        extra_layout = QtWidgets.QVBoxLayout(self.more_options)
        extra_layout.setContentsMargins(0, 0, 0, 0)
        extra_layout.setSpacing(10)
        extra_layout.addWidget(flags_box)

        self.ed_extra = QtWidgets.QLineEdit(spec.extra_flags)
        self.ed_extra.setPlaceholderText("Optional additional CatGT arguments")
        extra_layout.addWidget(QtWidgets.QLabel("Extra CatGT flags"))
        extra_layout.addWidget(self.ed_extra)
        main.addWidget(self.more_options)
        self.more_options.hide()
        self.btn_more_options.toggled.connect(self._toggle_more_options)

        self.preview = QtWidgets.QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(84)
        self.preview_label = QtWidgets.QLabel("Generated command fragment")
        main.addWidget(self.preview_label)
        main.addWidget(self.preview)

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
            self.ck_lfp_filter,
            self.sp_lfp_lowpass,
            self.cb_lfp_downsample,
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
        self.ck_lfp_filter.stateChanged.connect(self._sync_enabled_state)
        self.ck_gfix.stateChanged.connect(self._sync_enabled_state)
        self._sync_enabled_state()
        self._refresh_preview()

    def _toggle_more_options(self, expanded: bool) -> None:
        """Keep uncommon processing arguments available without crowding the filters."""
        self.more_options.setVisible(expanded)
        self.btn_more_options.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)

    def _reflow_filter_cards(self, width: int) -> None:
        """Stack filter cards when three readable columns no longer fit."""
        columns = 3 if width >= 820 else 1
        if columns == self._card_columns:
            return
        self._card_columns = columns
        for card in self._cards:
            self.filter_grid.removeWidget(card)
        for column in range(3):
            self.filter_grid.setColumnStretch(column, 1 if column < columns else 0)
        for index, card in enumerate(self._cards):
            self.filter_grid.addWidget(card, index // columns, index % columns)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Adapt the filter layout to the available tab width."""
        super().resizeEvent(event)
        self._reflow_filter_cards(event.size().width())

    def _sync_enabled_state(self) -> None:
        ap_enabled = self.ck_ap_filter.isChecked()
        for widget in [self.cb_ap_type, self.sp_ap_order, self.sp_ap_high, self.sp_ap_low]:
            widget.setEnabled(ap_enabled)
        gfix_enabled = self.ck_gfix.isChecked()
        for widget in [self.sp_gfix_amp, self.sp_gfix_slope, self.sp_gfix_noise]:
            widget.setEnabled(gfix_enabled)
        lfp_enabled = self.ck_lfp_filter.isChecked()
        for widget in [self.sp_lfp_lowpass, self.cb_lfp_downsample]:
            widget.setEnabled(lfp_enabled)

    def _refresh_preview(self) -> None:
        value = build_catgt_command_string(self.spec())
        self.preview.setPlainText(value)
        self.commandChanged.emit(value)

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
            use_lfp_filter=self.ck_lfp_filter.isChecked(),
            lfp_lowpass_hz=float(self.sp_lfp_lowpass.value()),
            lfp_downsample=int(self.cb_lfp_downsample.currentText()),
            use_gfix=self.ck_gfix.isChecked(),
            gfix_amp_mv=float(self.sp_gfix_amp.value()),
            gfix_slope_mv_per_sample=float(self.sp_gfix_slope.value()),
            gfix_noise_mv=float(self.sp_gfix_noise.value()),
            extra_flags=self.ed_extra.text().strip(),
        )

    def value(self) -> str:
        """Return the generated CatGT command fragment for the current form state."""
        return build_catgt_command_string(self.spec())


class CatGTStringBuilderDialog(QtWidgets.QDialog):
    """Compatibility dialog wrapper around :class:`CatGTStringBuilderPanel`."""

    def __init__(self, initial_command: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Build CatGT command string")
        self.resize(1120, 680)
        layout = QtWidgets.QVBoxLayout(self)
        self.panel = CatGTStringBuilderPanel(initial_command, self)
        layout.addWidget(self.panel, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def value(self) -> str:
        """Return the generated CatGT command fragment."""
        return self.panel.value()


class TPrimeStringBuilderPanel(QtWidgets.QWidget):
    """Reusable TPrime and CatGT event editor for standalone or combined dialogs."""

    valuesChanged = QtCore.Signal(str, str)

    def __init__(
        self,
        initial_to_stream: str,
        initial_extractors: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
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
        self.valuesChanged.emit(to_stream, ex_string)

    def values(self) -> Tuple[str, str]:
        """Return (toStream_sync_params, extractor_string) for the current editor state."""
        specs: List[TPrimeExtractorSpec] = []
        for row in range(self.tbl.rowCount()):
            spec = self._row_spec(row)
            if spec is not None:
                specs.append(spec)
        to_stream = build_tostream_sync_params(self.cb_stream_kind.currentText(), int(self.sp_stream_index.value()))
        extractors = build_tprime_extractor_string(specs, self.ed_extra.text().strip())
        return to_stream, extractors


class TPrimeStringBuilderDialog(QtWidgets.QDialog):
    """Compatibility dialog wrapper around :class:`TPrimeStringBuilderPanel`."""

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
        layout = QtWidgets.QVBoxLayout(self)
        self.panel = TPrimeStringBuilderPanel(initial_to_stream, initial_extractors, self)
        layout.addWidget(self.panel, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> Tuple[str, str]:
        """Return the built TPrime reference stream and CatGT extractor flags."""
        return self.panel.values()


class CatGTSetupDialog(QtWidgets.QDialog):
    """One guided CatGT setup window for run selection, filtering, events, and preview."""

    def __init__(
        self,
        *,
        initial_command: str,
        initial_to_stream: str,
        initial_extractors: str,
        initial_output_streams: str,
        initial_gate: str,
        initial_trigger: str,
        initial_probe: str,
        initial_probe_count: int,
        initial_probe_ids: str,
        initial_lf_lowpass_hz: float,
        initial_lf_downsample: int,
        initial_car_mode: str,
        initial_loccar_min_um: float,
        initial_loccar_max_um: float,
        run_name: str = "",
        input_directory: str = "",
        output_directory: str = "",
        current_gate: str = "0",
        current_trigger: str = "0,0",
        current_probe: str = "0",
        all_gate_range: str = "0,0",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("compactDialog", True)
        self.setWindowTitle("CatGT setup")
        self.setObjectName("CatGTSetupDialog")
        self.setMinimumSize(720, 580)
        screen = self.screen().availableGeometry()
        self.resize(min(1120, screen.width() - 40), min(880, screen.height() - 60))
        self.run_name = str(run_name).strip()
        self.input_directory = str(input_directory).strip()
        self.output_directory = str(output_directory).strip()
        self.current_gate = str(current_gate).strip() or "0"
        self.current_trigger = str(current_trigger).strip() or "0,0"
        self.current_probe = str(current_probe).strip() or "0"
        self.all_gate_range = str(all_gate_range).strip() or "0,0"

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        intro = QtWidgets.QLabel(
            "Set which part of the recording CatGT should process, choose AP or LFP output, and configure filtering "
            "and event extraction. The command preview updates as you edit."
        )
        intro.setObjectName("SectionHint")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._build_run_selection_tab(initial_gate, initial_trigger, initial_probe, initial_probe_count, initial_probe_ids)
        self._build_processing_tab(
            " ".join(
                token for token in _split_flags(initial_command)
                if not re.match(r"^-(?:xd|xid|xa|xia)=", re.sub(r"\[[^\]]*\]$", "", token), re.IGNORECASE)
            ),
            initial_output_streams,
            initial_lf_lowpass_hz,
            initial_lf_downsample,
            initial_car_mode,
            initial_loccar_min_um,
            initial_loccar_max_um,
        )
        self.event_panel = TPrimeStringBuilderPanel(initial_to_stream, initial_extractors, self)
        event_page = QtWidgets.QWidget()
        event_layout = QtWidgets.QVBoxLayout(event_page)
        event_layout.setContentsMargins(20, 20, 20, 20)
        event_hint = QtWidgets.QLabel(
            "Define digital or analog edges for CatGT. The TPrime reference stream is shown separately in the preview. "
            "Use the extra flags field for CatGT bit-field (-bf) extraction."
        )
        event_hint.setObjectName("SectionHint")
        event_hint.setWordWrap(True)
        event_layout.addWidget(event_hint)
        event_layout.addWidget(self.event_panel, 1)
        self._add_scroll_tab(event_page, "Events and sync")

        preview_box = QtWidgets.QGroupBox("Full CatGT command preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(16, 20, 16, 12)
        preview_layout.setSpacing(8)
        self.command_preview = QtWidgets.QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self.command_preview.setWordWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.command_preview.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.command_preview.setFixedHeight(112)
        command_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.command_preview.setFont(command_font)
        self.command_preview.setStyleSheet(f"font-family: '{command_font.family()}'; font-size: 12px;")
        preview_layout.addWidget(self.command_preview)
        preview_footer = QtWidgets.QHBoxLayout()
        self.tprime_preview = QtWidgets.QLabel()
        self.tprime_preview.setObjectName("SectionHint")
        self.tprime_preview.setWordWrap(True)
        preview_footer.addWidget(self.tprime_preview, 1)
        self.btn_copy_command = QtWidgets.QPushButton("Copy command")
        self.btn_copy_command.setProperty("role", "ghost")
        self.btn_copy_command.clicked.connect(
            lambda: QtWidgets.QApplication.clipboard().setText(self.command_preview.toPlainText())
        )
        preview_footer.addWidget(self.btn_copy_command)
        preview_layout.addLayout(preview_footer)
        layout.addWidget(preview_box)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Apply settings")
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.catgt_panel.commandChanged.connect(self._refresh_command_preview)
        self.event_panel.valuesChanged.connect(self._refresh_command_preview_from_events)
        self._refresh_command_preview()

    def _add_scroll_tab(self, page: QtWidgets.QWidget, title: str) -> None:
        """Let settings scroll independently while preview and Apply remain accessible."""
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setMinimumSize(0, 0)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, title)

    @staticmethod
    def _field(title: str, editor: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Place a wrapping label above its editor with consistent spacing."""
        field = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(field)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QtWidgets.QLabel(title)
        label.setObjectName("FieldTitle")
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(editor)
        return field

    def _build_run_selection_tab(
        self,
        initial_gate: str,
        initial_trigger: str,
        initial_probe: str,
        initial_probe_count: int,
        initial_probe_ids: str,
    ) -> None:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setContentsMargins(24, 24, 24, 24)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(12)

        current_range = self._parse_pair(initial_gate, self.current_gate)
        self.cb_gate_mode = QtWidgets.QComboBox()
        self.cb_gate_mode.addItem("Use the gate in the input file", "current")
        self.cb_gate_mode.addItem("All gates in this run", "all")
        self.cb_gate_mode.addItem("Choose a gate range", "range")
        initial_gate_mode = "all" if str(initial_gate).lower() == "all" else (
            "range" if str(initial_gate).lower() not in {"current", ""} else "current"
        )
        self.cb_gate_mode.setCurrentIndex(max(0, self.cb_gate_mode.findData(initial_gate_mode)))
        self.sp_gate_first = self._number_spin(current_range[0], 9999)
        self.sp_gate_last = self._number_spin(current_range[1], 9999)
        form.addRow("Gate selection", self.cb_gate_mode)
        self.gate_range_row = self._pair_widget(self.sp_gate_first, self.sp_gate_last)
        form.addRow("First gate / last gate", self.gate_range_row)
        gate_hint = QtWidgets.QLabel(
            "Use the input file's gate for one gate, or let CatGT join all available gates in the run. "
            "Choose a range to set the first and last gate explicitly."
        )
        gate_hint.setObjectName("SectionHint")
        gate_hint.setWordWrap(True)
        form.addRow("", gate_hint)

        current_trigger_range = self._parse_pair(initial_trigger, self.current_trigger)
        self.cb_trigger_mode = QtWidgets.QComboBox()
        self.cb_trigger_mode.addItem("Use the trigger in the input file", "current")
        self.cb_trigger_mode.addItem("All triggers in this run", "all")
        self.cb_trigger_mode.addItem("Choose a trigger range", "range")
        initial_trigger_mode = "all" if str(initial_trigger).lower() == "all" else (
            "range" if str(initial_trigger).lower() not in {"current", ""} else "current"
        )
        self.cb_trigger_mode.setCurrentIndex(max(0, self.cb_trigger_mode.findData(initial_trigger_mode)))
        self.sp_trigger_first = self._number_spin(current_trigger_range[0], 999999)
        self.sp_trigger_last = self._number_spin(current_trigger_range[1], 999999)
        form.addRow("Trigger selection", self.cb_trigger_mode)
        self.trigger_range_row = self._pair_widget(self.sp_trigger_first, self.sp_trigger_last)
        form.addRow("First trigger / last trigger", self.trigger_range_row)
        self.selection_form = form
        trigger_hint = QtWidgets.QLabel(
            "All triggers maps to CatGT's start,end range. A selected range maps to the corresponding -t=first,last option."
        )
        trigger_hint.setObjectName("SectionHint")
        trigger_hint.setWordWrap(True)
        form.addRow("", trigger_hint)

        self.sp_probe_count = self._number_spin(max(1, int(initial_probe_count)), 32, minimum=1)
        self.ed_probe_ids = QtWidgets.QLineEdit(initial_probe_ids or "0")
        self.ed_probe_ids.setPlaceholderText("Example: 0-3 or 0,2,5")
        self.cb_probe_target = QtWidgets.QComboBox()
        expected_ids = "0" if self.sp_probe_count.value() == 1 else f"0-{self.sp_probe_count.value() - 1}"
        self._probe_ids_auto = not initial_probe_ids or str(initial_probe_ids).strip() == expected_ids
        self._refresh_probe_targets(str(initial_probe))
        form.addRow("Probes recorded at once", self.sp_probe_count)
        form.addRow("Recorded probe IDs", self.ed_probe_ids)
        form.addRow("Process this probe", self.cb_probe_target)
        probe_hint = QtWidgets.QLabel(
            "The sorter handles one probe per queued job. Choose the probe ID to process; the count helps populate the available IDs."
        )
        probe_hint.setObjectName("SectionHint")
        probe_hint.setWordWrap(True)
        form.addRow("", probe_hint)

        for combo in [self.cb_gate_mode, self.cb_trigger_mode, self.cb_probe_target]:
            combo.currentIndexChanged.connect(self._sync_selection_controls)
            combo.currentIndexChanged.connect(self._refresh_command_preview)
        for spin in [self.sp_gate_first, self.sp_gate_last, self.sp_trigger_first, self.sp_trigger_last]:
            spin.valueChanged.connect(self._refresh_command_preview)
        self.sp_probe_count.valueChanged.connect(self._on_probe_count_changed)
        self.ed_probe_ids.textEdited.connect(self._on_probe_ids_edited)
        self.ed_probe_ids.textChanged.connect(self._refresh_command_preview)
        self._sync_selection_controls()
        self._add_scroll_tab(page, "Run and probes")

    def _build_processing_tab(
        self,
        initial_command: str,
        initial_output_streams: str,
        initial_lf_lowpass_hz: float,
        initial_lf_downsample: int,
        initial_car_mode: str,
        initial_loccar_min_um: float,
        initial_loccar_max_um: float,
    ) -> None:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(20, 20, 20, 20)
        page_layout.setSpacing(16)
        output_box = QtWidgets.QGroupBox("Output and referencing")
        top = QtWidgets.QGridLayout(output_box)
        top.setContentsMargins(16, 20, 16, 16)
        top.setHorizontalSpacing(20)
        top.setVerticalSpacing(12)
        top.setColumnStretch(0, 1)
        top.setColumnStretch(1, 1)
        self.cb_output_streams = QtWidgets.QComboBox()
        for label, value in [("Action potentials (AP)", "ap"), ("Local field potential (LFP)", "lfp"), ("AP + LFP", "both")]:
            self.cb_output_streams.addItem(label, value)
        index = self.cb_output_streams.findData(initial_output_streams)
        self.cb_output_streams.setCurrentIndex(index if index >= 0 else 0)
        self.cb_car_mode = QtWidgets.QComboBox()
        for label, value in [
            ("Global, demultiplexed (gbldmx)", "gbldmx"),
            ("Global average (gblcar)", "gblcar"),
            ("Local average (loccar)", "loccar"),
            ("No referencing", "none"),
        ]:
            self.cb_car_mode.addItem(label, value)
        self.cb_car_mode.setCurrentIndex(max(0, self.cb_car_mode.findData(initial_car_mode)))
        self.sp_loccar_min = QtWidgets.QDoubleSpinBox()
        self.sp_loccar_min.setRange(10.0, 500.0)
        self.sp_loccar_min.setDecimals(1)
        self.sp_loccar_min.setValue(float(initial_loccar_min_um))
        self.sp_loccar_max = QtWidgets.QDoubleSpinBox()
        self.sp_loccar_max.setRange(10.0, 1000.0)
        self.sp_loccar_max.setDecimals(1)
        self.sp_loccar_max.setValue(float(initial_loccar_max_um))
        top.addWidget(self._field("Neural output", self.cb_output_streams), 0, 0)
        top.addWidget(self._field("Common average reference", self.cb_car_mode), 0, 1)
        self.local_reference_field = self._field(
            "Local reference radius (µm): inner to outer",
            self._pair_widget(self.sp_loccar_min, self.sp_loccar_max),
        )
        top.addWidget(self.local_reference_field, 1, 0, 1, 2)
        page_layout.addWidget(output_box)

        self.catgt_panel = CatGTStringBuilderPanel(initial_command, self)
        # The persistent full preview replaces the panel's standalone fragment preview.
        self.catgt_panel.preview.hide()
        self.catgt_panel.preview_label.hide()
        self.catgt_panel.sp_lfp_lowpass.setValue(initial_lf_lowpass_hz)
        self.catgt_panel.cb_lfp_downsample.setCurrentText(str(initial_lf_downsample))
        page_layout.addWidget(self.catgt_panel)
        page_layout.addStretch(1)
        self.cb_output_streams.currentIndexChanged.connect(self._sync_output_controls)
        self.cb_car_mode.currentIndexChanged.connect(self._sync_reference_controls)
        self.cb_car_mode.currentTextChanged.connect(self._refresh_command_preview)
        self.sp_loccar_min.valueChanged.connect(self._refresh_command_preview)
        self.sp_loccar_max.valueChanged.connect(self._refresh_command_preview)
        self._sync_output_controls()
        self._sync_reference_controls()
        self._add_scroll_tab(page, "AP/LFP processing")

    @staticmethod
    def _number_spin(value: int, maximum: int, minimum: int = 0) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(max(minimum, min(maximum, int(value))))
        return spin

    @staticmethod
    def _pair_widget(first: QtWidgets.QWidget, last: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Give numeric ranges enough room to show their values and spin controls."""
        first.setMinimumWidth(110)
        last.setMinimumWidth(110)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(first)
        row.addWidget(QtWidgets.QLabel("through"))
        row.addWidget(last)
        row.addStretch(1)
        widget = QtWidgets.QWidget()
        widget.setLayout(row)
        return widget

    @staticmethod
    def _parse_pair(raw: str, fallback: str) -> Tuple[int, int]:
        text = str(raw).strip()
        if text.lower() in {"current", "all", "start,end"}:
            text = fallback
        try:
            values = [int(part.strip()) for part in text.split(",", 1)]
            return (values[0], values[-1])
        except (TypeError, ValueError):
            return (0, 0)

    def _on_probe_count_changed(self, count: int) -> None:
        if self._probe_ids_auto:
            self.ed_probe_ids.setText("0" if count == 1 else f"0-{count - 1}")
        self._refresh_probe_targets()

    def _on_probe_ids_edited(self, _text: str) -> None:
        self._probe_ids_auto = False
        self._refresh_probe_targets()

    def _refresh_probe_targets(self, preferred: str = "current") -> None:
        if hasattr(self, "cb_probe_target"):
            previous = str(self.cb_probe_target.currentData() or preferred or "current")
            self.cb_probe_target.blockSignals(True)
            self.cb_probe_target.clear()
            self.cb_probe_target.addItem("Use probe from input file", "current")
        else:
            previous = str(preferred or "current")
            self.cb_probe_target = QtWidgets.QComboBox()
            self.cb_probe_target.addItem("Use probe from input file", "current")
        try:
            probe_ids = parse_channel_spec(self.ed_probe_ids.text().strip())
        except (TypeError, ValueError):
            probe_ids = []
        for probe_id in probe_ids:
            self.cb_probe_target.addItem(f"Probe {probe_id}", str(probe_id))
        index = self.cb_probe_target.findData(previous)
        if index < 0 and preferred not in {"", "current"}:
            index = self.cb_probe_target.findData(str(preferred))
        self.cb_probe_target.setCurrentIndex(index if index >= 0 else 0)
        self.cb_probe_target.blockSignals(False)
        self._refresh_command_preview()

    def _sync_selection_controls(self, *_args) -> None:
        gate_custom = self.cb_gate_mode.currentData() == "range"
        self.sp_gate_first.setEnabled(gate_custom)
        self.sp_gate_last.setEnabled(gate_custom)
        trigger_custom = self.cb_trigger_mode.currentData() == "range"
        self.sp_trigger_first.setEnabled(trigger_custom)
        self.sp_trigger_last.setEnabled(trigger_custom)
        self.selection_form.setRowVisible(self.gate_range_row, gate_custom)
        self.selection_form.setRowVisible(self.trigger_range_row, trigger_custom)

    def _sync_reference_controls(self, *_args) -> None:
        """Show local radius controls only when local referencing is selected."""
        self.local_reference_field.setVisible(self.cb_car_mode.currentData() == "loccar")

    def _sync_output_controls(self, *_args) -> None:
        output_streams = str(self.cb_output_streams.currentData() or "ap")
        use_lfp = output_streams in {"lfp", "both"}
        self.catgt_panel.ck_lfp_filter.setChecked(use_lfp)
        self.catgt_panel.ck_lfp_filter.setEnabled(use_lfp)
        self.catgt_panel._sync_enabled_state()
        self._refresh_command_preview()

    def _resolved_gate(self) -> str:
        mode = str(self.cb_gate_mode.currentData())
        if mode == "all":
            return self.all_gate_range
        if mode == "range":
            return f"{self.sp_gate_first.value()},{self.sp_gate_last.value()}"
        return self.current_gate

    def _resolved_trigger(self) -> str:
        mode = str(self.cb_trigger_mode.currentData())
        if mode == "all":
            return "start,end"
        if mode == "range":
            return f"{self.sp_trigger_first.value()},{self.sp_trigger_last.value()}"
        return self.current_trigger

    def _refresh_command_preview_from_events(self, *_args) -> None:
        self._refresh_command_preview()

    def _refresh_command_preview(self, *_args) -> None:
        if not hasattr(self, "command_preview"):
            return
        to_stream, event_flags = self.event_panel.values() if hasattr(self, "event_panel") else ("imec0", "")
        selected_probe = str(self.cb_probe_target.currentData() or "current")
        self.command_preview.setPlainText(
            build_catgt_command_preview(
                run_name=self.run_name,
                input_directory=self.input_directory,
                output_directory=self.output_directory,
                gate_string=self._resolved_gate(),
                trigger_string=self._resolved_trigger(),
                probe_string=self.current_probe if selected_probe == "current" else selected_probe,
                output_streams=str(self.cb_output_streams.currentData() or "ap"),
                car_mode=str(self.cb_car_mode.currentData()),
                loccar_min_um=float(self.sp_loccar_min.value()),
                loccar_max_um=float(self.sp_loccar_max.value()),
                command_flags=self.catgt_panel.value(),
                extractor_flags=event_flags,
            )
        )
        self.tprime_preview.setText(f"TPrime reference stream: {to_stream}")

    def _accept_if_valid(self) -> None:
        try:
            ids = parse_channel_spec(self.ed_probe_ids.text().strip())
            if not ids:
                raise ValueError("Enter at least one recorded probe ID.")
            selected_probe = str(self.cb_probe_target.currentData() or "current")
            if selected_probe not in {"current", *[str(probe_id) for probe_id in ids]}:
                raise ValueError("The selected probe is not in the recorded probe IDs list.")
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Check run selection", str(exc))
            return
        self.accept()

    def values(self) -> dict[str, object]:
        """Return generated GUI settings and CatGT selectors for the preprocessing form."""
        to_stream, extractors = self.event_panel.values()
        gate_mode = str(self.cb_gate_mode.currentData())
        trigger_mode = str(self.cb_trigger_mode.currentData())
        gate = "current" if gate_mode == "current" else ("all" if gate_mode == "all" else self._resolved_gate())
        trigger = "current" if trigger_mode == "current" else ("all" if trigger_mode == "all" else self._resolved_trigger())
        return {
            "catgt_cmd_string": self.catgt_panel.value(),
            "ni_extract_string": extractors,
            "tostream_sync_params": to_stream,
            "catgt_output_streams": str(self.cb_output_streams.currentData() or "ap"),
            "catgt_lf_lowpass_hz": float(self.catgt_panel.sp_lfp_lowpass.value()),
            "catgt_lf_downsample": int(self.catgt_panel.cb_lfp_downsample.currentText()),
            "catgt_car_mode": str(self.cb_car_mode.currentData()),
            "catgt_loccar_min_um": float(self.sp_loccar_min.value()),
            "catgt_loccar_max_um": float(self.sp_loccar_max.value()),
            "gate_string": gate,
            "trigger_string": trigger,
            "probe_string": str(self.cb_probe_target.currentData() or "current"),
            "probe_count": int(self.sp_probe_count.value()),
            "probe_ids": self.ed_probe_ids.text().strip(),
        }
