"""Part 2 PyQt6 frontend: Speech → VSL structure → 3D avatar.

Mirrors the spec's two-column Streamlit layout on desktop:

    Left : microphone button, transcript, VSL structure chips
    Right: keypoint-driven 3D avatar viewport + playback controls
    Bottom: chat history with per-item confidence
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_PART2 = Path(__file__).resolve().parents[1]

# make `backend`, `avatar`, `frontend` importable regardless of cwd
for sub in ("backend", "avatar", "frontend"):
    sys.path.insert(0, str(_PART2))

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def _load(alias_top: str, real: str):
    """Import ``alias_top.<rest>`` if the alias is registered, else ``real``."""
    import sys as _sys
    from importlib import import_module

    if alias_top in _sys.modules:
        return import_module(f"{alias_top}.{real.split('.', 1)[1]}")
    return import_module(real)


AvatarViewport = _load("p2avatar", "avatar.view").AvatarViewport
_sr = _load("p2backend", "backend.speech_recognition")
STTUnavailableError = _sr.STTUnavailableError
write_wav = _sr.write_wav
process_audio = _load("p2frontend", "frontend.pipeline").process_audio
process_text = _load("p2frontend", "frontend.pipeline").process_text
MicrophoneRecorder = _load("p2frontend", "frontend.recorder").MicrophoneRecorder


class _ProcessWorker(QThread):
    """Runs the speech pipeline off the UI thread; emits results."""

    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, task: dict) -> None:
        super().__init__()
        self._task = task

    def run(self) -> None:
        try:
            if self._task["kind"] == "audio":
                result = process_audio(self._task["samples"], self._task["rate"])
            else:
                result = process_text(self._task["text"])
            self.done.emit(result)
        except STTUnavailableError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:                    # pragma: no cover - defensive
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class SpeechToSignApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Avatar 3D - Speech to Sign (VSL)")
        self.resize(1280, 820)
        self.setStyleSheet(_STYLE)

        central = QWidget()
        root = QVBoxLayout(central)

        body = QHBoxLayout()

        left = QVBoxLayout()

        # mic / input
        self.record_btn = QPushButton("●  Bắt đầu nói")
        self.record_btn.setObjectName("primary")
        self.record_btn.clicked.connect(self._toggle_record)
        left.addWidget(self.record_btn)

        self.input_row = QHBoxLayout()
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("hoặc gõ câu tiếng Việt (demo offline)...")
        self.text_input.returnPressed.connect(self._send_text)
        send_btn = QPushButton("Gửi")
        send_btn.clicked.connect(self._send_text)
        self.input_row.addWidget(self.text_input, 1)
        self.input_row.addWidget(send_btn)
        left.addLayout(self.input_row)

        self.status = QLabel("Nhấn 'Bắt đầu nói' hoặc nhập câu để bắt đầu.")
        self.status.setObjectName("subtitle")
        self.status.setWordWrap(True)
        left.addWidget(self.status)

        # transcript
        transcript_gb = QGroupBox("Văn bản nhận dạng")
        tv = QVBoxLayout(transcript_gb)
        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        tv.addWidget(self.transcript)
        left.addWidget(transcript_gb)

        # VSL structure
        struct_gb = QGroupBox("Cấu trúc VSL")
        sv = QVBoxLayout(struct_gb)
        self.topic_label = QLabel("topic: —")
        self.comment_label = QLabel("comment: —")
        self.words_label = QLabel("words: —")
        self.words_label.setWordWrap(True)
        for lbl in (self.topic_label, self.comment_label, self.words_label):
            lbl.setObjectName("stat")
            sv.addWidget(lbl)
        left.addWidget(struct_gb)

        body.addLayout(left, 1)

        right = QVBoxLayout()
        self.viewport = AvatarViewport()
        right.addWidget(self.viewport, 3)
        body.addLayout(right, 1)

        root.addLayout(body, 1)

        history_gb = QGroupBox("Lịch sử hội thoại")
        hv = QVBoxLayout(history_gb)
        self.history = QListWidget()
        hv.addWidget(self.history)
        root.addWidget(history_gb)

        self.setCentralWidget(central)

        self._recorder = MicrophoneRecorder()
        self._recording = False
        self._worker: _ProcessWorker | None = None

    # -- actions -------------------------------------------------------------
    def _toggle_record(self) -> None:
        if self._recording:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self) -> None:
        try:
            self._recorder.start()
        except Exception as exc:
            self.status.setText(f"Không mở được microphone: {exc}")
            return
        self._recording = True
        self.record_btn.setText("■  Dừng")
        self.status.setText("Đang nghe… hãy nói tiếng Việt.")

    def _stop_record(self) -> None:
        samples = self._recorder.stop()
        self._recording = False
        self.record_btn.setText("●  Bắt đầu nói")
        if len(samples) < 1600:
            self.status.setText("Không thu được âm thanh (quá ngắn).")
            return
        self.status.setText("Đang nhận dạng…")
        self._dispatch({"kind": "audio", "samples": samples,
                        "rate": MicrophoneRecorder.SAMPLE_RATE})

    def _send_text(self) -> None:
        text = self.text_input.text().strip()
        if not text:
            return
        self.text_input.clear()
        self.status.setText("Đang dịch sang VSL…")
        self.transcript.setText(text)
        self._dispatch({"kind": "text", "text": text})

    def _dispatch(self, task: dict) -> None:
        self._worker = _ProcessWorker(task)
        self._worker.done.connect(self._on_result)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    # -- results -------------------------------------------------------------
    def _on_result(self, result: dict) -> None:
        structure = result["vsl_structure"]
        self.topic_label.setText(f"topic: {structure['topic']}")
        self.comment_label.setText(f"comment: {structure['comment']}")
        self.words_label.setText("words: " + " → ".join(structure["words"]))
        missing = result.get("missing_animation") or []
        status = f"Xong • độ tin cậy {result['confidence']:.2f}"
        if missing:
            status += (" • ⚠ không có dữ liệu hình thể: "
                       + ", ".join(dict.fromkeys(missing)))
        self.status.setText(status)

        steps = result["animation_sequence"]
        clips = []
        for feature in result["clips"]:
            if feature is None:
                continue
            clips.append(np.asarray(feature, dtype=np.float32))
        if clips:
            fps = result["controls"]["fps"]
            self.viewport.play_clips(clips, fps)
            self.viewport.play()
        else:
            self.status.setText(
                "Không có ký hiệu nào phát được (dữ liệu đang rỗng). "
                "Thử một câu khác.")
            return

        shown = " ".join(structure["words"]) or structure["comment"]
        self.history.insertItem(
            0, f"[{result['confidence']:.2f}] {result['original_text']} → {shown}")

    def _on_failed(self, message: str) -> None:
        if "Unavailable" in message or "model not loaded" in message:
            self.status.setText(
                "Whisper chưa khả dụng (chưa có mô hình/offline). "
                "Bạn có thể gõ câu trực tiếp để xem avatar.")
        else:
            self.status.setText(f"Lỗi: {message}")

    def stop(self) -> None:
        if self._recording:
            self._recorder.stop()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(3000)


_STYLE = """
QMainWindow { background: #101216; }
QLabel#title { font-size: 22px; font-weight: 600; color: #e8e8e8; }
QLabel#subtitle { color: #8a8f98; }
QLabel#stat { color: #c7ccd4; }
QPushButton { background: #1f242e; color: #e8e8e8; border: 1px solid #333a46;
              border-radius: 8px; padding: 8px 14px; }
QPushButton:hover { background: #2b3240; }
QPushButton#primary { background: #2d6cdf; border: none; color: white; }
QGroupBox { border: 1px solid #262c36; border-radius: 8px; margin-top: 10px;
            padding-top: 8px; color: #c7ccd4; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QTextEdit, QLineEdit, QListWidget { background: #171b22; color: #e8e8e8;
           border: 1px solid #262c36; border-radius: 6px; }
"""


def run() -> None:
    app = QApplication(sys.argv)
    win = SpeechToSignApp()
    win.show()
    app.aboutToQuit.connect(win.stop)
    sys.exit(app.exec())


if __name__ == "__main__":
    run()