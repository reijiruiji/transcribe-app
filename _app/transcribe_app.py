"""
文字起こしアプリ v2.1 — WhisperX + pyannote.audio
完全ローカル処理・話者分離・GPU対応
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import gc
import json
import subprocess
import webbrowser
from pathlib import Path
from datetime import datetime

CONFIG_FILE = Path(__file__).parent / "transcribe_config.json"
OUTPUT_DIR  = Path(__file__).parent.parent / "文字起こし"

APP_VERSION = "v2.1"

# ===== 設定 =====
def load_config():
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"hf_token": "", "output_dir": str(OUTPUT_DIR), "model": "large-v3", "device": "auto"}

def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

# ===== GPU 検出 =====
def detect_gpu():
    """Returns (has_cuda, gpu_name, torch_version)"""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0), torch.__version__
        return False, None, torch.__version__
    except ImportError:
        return False, None, "未インストール"

def resolve_device(preferred: str):
    has_cuda, _, _ = detect_gpu()
    if preferred == "auto":
        return "cuda" if has_cuda else "cpu"
    if preferred == "cuda" and not has_cuda:
        return "cpu"
    return preferred

# ===== 文字起こし処理 =====
def run_transcription(input_file, output_dir, hf_token, model_size, device_pref, log_q):

    def log(msg):    log_q.put(("log", msg))
    def done(path):  log_q.put(("done", path))
    def error(msg):  log_q.put(("error", msg))

    try:
        import whisperx
        import whisperx.diarize

        device = resolve_device(device_pref)
        compute_type = "float16" if device == "cuda" else "int8"
        batch = 16 if device == "cuda" else 4

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        has_cuda, gpu_name, _ = detect_gpu()
        device_disp = f"GPU ({gpu_name})" if device == "cuda" else "CPU（GPUより遅い）"
        log(f"  ファイル : {Path(input_file).name}")
        log(f"  処理方式 : {device_disp}")
        log(f"  モデル   : {model_size}")
        log("")

        log("【1/4】 音声を読み込んでいます...")
        audio = whisperx.load_audio(input_file)

        log("【2/4】 文字起こし中...")
        model = whisperx.load_model(model_size, device, compute_type=compute_type, language="ja")
        result = model.transcribe(audio, batch_size=batch)
        del model; gc.collect()
        log("       完了")

        log("【3/4】 タイムスタンプを合わせています...")
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device)
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device,
            return_char_alignments=False)
        del align_model; gc.collect()
        log("       完了")

        log("【4/4】 発言者を分析中...")
        diarize = whisperx.diarize.DiarizationPipeline(token=hf_token, device=device)
        diarize_segments = diarize(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        log("       完了")

        stem = Path(input_file).stem
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_txt  = Path(output_dir) / f"{stem}_{ts}.txt"
        out_json = Path(output_dir) / f"{stem}_{ts}.json"

        def fmt(sec):
            h, m = int(sec // 3600), int((sec % 3600) // 60)
            s = sec % 60
            return f"{h:02d}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"

        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"元ファイル : {input_file}\n")
            f.write(f"作成日時   : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"処理方式   : {device_disp}\n")
            f.write("=" * 60 + "\n\n")
            cur = None
            for seg in result["segments"]:
                spk  = seg.get("speaker", "発言者??")
                text = seg["text"].strip()
                if not text:
                    continue
                if spk != cur:
                    f.write(f"\n【{spk}】\n"); cur = spk
                f.write(f"  {fmt(seg['start'])} - {fmt(seg['end'])}  {text}\n")

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result["segments"], f, ensure_ascii=False, indent=2)

        log("")
        log("✅ 完了しました！")
        log(f"   保存先: {out_txt}")
        done(str(out_txt))

    except ImportError as e:
        error(
            "【セットアップ未完了】必要なライブラリがインストールされていません。\n\n"
            "解決方法：\n"
            "① このアプリを閉じる\n"
            "② フォルダ内の「★セットアップ（初回だけここをダブルクリック）.bat」を\n"
            "   ダブルクリックする\n"
            "③ 黒い画面が閉じたらこのアプリを再起動する\n\n"
            f"（技術詳細: {e}）"
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        msg = _friendly_error(type(e).__name__, str(e), tb)
        error(msg)


# ===== エラーメッセージの日本語化 =====
def _friendly_error(etype: str, msg: str, tb: str) -> str:
    msg_lower = msg.lower()
    if "401" in msg or "unauthorized" in msg_lower or "token" in msg_lower and "invalid" in msg_lower:
        return (
            "【トークンエラー】HuggingFaceトークンが無効です。\n\n"
            "解決方法：\n"
            "① STEP 1のトークンを確認してください\n"
            "② hf_ で始まる文字列になっているか確認\n"
            "③ HuggingFaceで新しいトークンを発行して貼り直す\n\n"
            f"（技術詳細: {etype}: {msg}）"
        )
    if "403" in msg or "gated" in msg_lower or "access" in msg_lower and "denied" in msg_lower:
        return (
            "【モデルアクセス拒否】HuggingFaceのモデル利用規約への同意が必要です。\n\n"
            "解決方法：\n"
            "ブラウザで以下の2つのページを開いて「Agree」を押してください：\n"
            "① https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "② https://huggingface.co/pyannote/segmentation-3.0\n\n"
            f"（技術詳細: {etype}: {msg}）"
        )
    if "cuda" in msg_lower and ("out of memory" in msg_lower or "oom" in msg_lower):
        return (
            "【GPUメモリ不足】GPUのメモリ（VRAM）が足りません。\n\n"
            "解決方法：\n"
            "① 処理方式を「CPU」に変更して再試行\n"
            "② 他のアプリを閉じてからGPUで再試行\n\n"
            f"（技術詳細: {etype}: {msg}）"
        )
    if "filenotfounderror" in etype.lower() or "no such file" in msg_lower:
        return (
            "【ファイルが見つかりません】選択したファイルのパスが無効です。\n\n"
            "解決方法：\n"
            "「ファイルを選ぶ」ボタンでファイルを選び直してください。\n\n"
            f"（技術詳細: {etype}: {msg}）"
        )
    if "ffmpeg" in msg_lower:
        return (
            "【FFmpegが見つかりません】動画変換に必要なFFmpegがインストールされていません。\n\n"
            "解決方法：\n"
            "「★セットアップ.bat」を再実行してFFmpegをインストールしてください。\n\n"
            f"（技術詳細: {etype}: {msg}）"
        )
    return f"予期しないエラーが発生しました。\n\n{etype}: {msg}\n\n{tb}"


# ===== セットアップ確認 =====
def check_setup() -> tuple[bool, str]:
    """whisperxがインストール済みかチェック。(ok, message)を返す"""
    try:
        import whisperx  # noqa: F401
        return True, ""
    except ImportError:
        return False, (
            "whisperxがインストールされていません。\n"
            "「★セットアップ（初回だけここをダブルクリック）.bat」を実行してください。"
        )


# ===== HF トークン取得ガイドのポップアップ =====
def show_token_guide(parent):
    win = tk.Toplevel(parent)
    win.title("HuggingFaceトークンの取得方法")
    win.configure(bg="#1a1a2e")
    win.resizable(False, False)
    win.grab_set()

    BG, BG2 = "#1a1a2e", "#16213e"
    FG, ACCENT = "#e0e0e0", "#4fc3f7"
    YELLOW = "#ffd54f"

    tk.Label(win, text="HuggingFace トークンの取得手順",
             bg=BG, fg=ACCENT, font=("Yu Gothic UI", 13, "bold")).pack(padx=20, pady=(16, 4))

    steps = [
        ("STEP 1", "下のボタンから HuggingFace にアクセス\n（無料アカウントが必要です）"),
        ("STEP 2", "ログイン後、右上のアイコン →\n「Settings」→「Access Tokens」を開く"),
        ("STEP 3", "「New token」をクリック\n名前は何でもOK（例: transcribe）"),
        ("STEP 4", "権限は以下の1つだけにチェック：\n\n✅  Read access to contents of all\n      public gated repos you can access\n\n他は全部チェックなし"),
        ("STEP 5", "「Create token」を押してトークンをコピー\n（hf_で始まる文字列）"),
        ("STEP 6", "このアプリのSTEP 1欄に貼り付ける\n（次回から自動入力されます）"),
    ]

    for label, text in steps:
        row = tk.Frame(win, bg=BG2, relief="flat")
        row.pack(fill="x", padx=16, pady=3)
        tk.Label(row, text=f" {label} ", bg=ACCENT, fg="#1a1a2e",
                 font=("Yu Gothic UI", 8, "bold"), padx=6, pady=2).pack(side="left", padx=(10, 8), pady=8)
        tk.Label(row, text=text, bg=BG2, fg=FG, font=("Yu Gothic UI", 9),
                 justify="left", anchor="w").pack(side="left", pady=6)

    note = tk.Frame(win, bg="#2a1a0e")
    note.pack(fill="x", padx=16, pady=(8, 4))
    tk.Label(note, text="⚠  STEP 4 の権限1つだけで十分です。他の権限は不要。",
             bg="#2a1a0e", fg=YELLOW, font=("Yu Gothic UI", 9, "bold"),
             padx=10, pady=6).pack(anchor="w")

    tk.Button(win, text="HuggingFace を開く →",
              bg=ACCENT, fg="#1a1a2e", font=("Yu Gothic UI", 10, "bold"),
              relief="flat", padx=16, pady=8, cursor="hand2",
              command=lambda: webbrowser.open("https://huggingface.co/settings/tokens")
              ).pack(pady=(8, 4))

    tk.Button(win, text="閉じる",
              bg="#313244", fg=FG, font=("Yu Gothic UI", 9),
              relief="flat", padx=16, pady=6, cursor="hand2",
              command=win.destroy).pack(pady=(0, 16))


# ===== GPU 有効化ガイドのポップアップ =====
def show_gpu_guide(parent):
    win = tk.Toplevel(parent)
    win.title("GPU を有効にする")
    win.configure(bg="#1a1a2e")
    win.resizable(False, False)
    win.grab_set()

    BG, BG2 = "#1a1a2e", "#16213e"
    FG, ACCENT = "#e0e0e0", "#4fc3f7"
    GREEN, YELLOW = "#69f0ae", "#ffd54f"

    tk.Label(win, text="GPU を有効にする（CUDA版PyTorchのインストール）",
             bg=BG, fg=ACCENT, font=("Yu Gothic UI", 13, "bold")).pack(padx=20, pady=(16, 4))

    tk.Label(win, text="現在CPU版のPyTorchが入っています。\n以下のコマンドを実行するとGPUが使えるようになります。",
             bg=BG, fg=FG, font=("Yu Gothic UI", 10), justify="left").pack(padx=20, pady=(0, 10))

    cmd = "py -3.12 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128"

    code_frame = tk.Frame(win, bg="#0d1117")
    code_frame.pack(fill="x", padx=20, pady=4)
    tk.Label(code_frame, text=cmd, bg="#0d1117", fg=GREEN,
             font=("Consolas", 8), padx=10, pady=10, wraplength=480,
             justify="left").pack(anchor="w")

    def copy_cmd():
        win.clipboard_clear()
        win.clipboard_append(cmd)
        copy_btn.config(text="コピーしました ✓")
        win.after(2000, lambda: copy_btn.config(text="コマンドをコピー"))

    copy_btn = tk.Button(win, text="コマンドをコピー",
                         bg="#313244", fg=FG, font=("Yu Gothic UI", 9),
                         relief="flat", padx=12, pady=6, cursor="hand2",
                         command=copy_cmd)
    copy_btn.pack(pady=(4, 0))

    tk.Label(win,
             text="手順：\n"
                  "① 今の文字起こしが終わるまで待つ\n"
                  "② スタートメニュー → 「コマンドプロンプト」を開く\n"
                  "③ 上のコマンドを貼り付けてEnter（約3GB、10〜20分）\n"
                  "④ 完了後にアプリを再起動すると自動でGPUが選ばれます",
             bg=BG2, fg=FG, font=("Yu Gothic UI", 9),
             justify="left", padx=14, pady=10).pack(fill="x", padx=20, pady=10)

    tk.Label(win, text="GPU使用時: 処理速度が約5〜10倍速くなります",
             bg=BG, fg=YELLOW, font=("Yu Gothic UI", 9, "bold")).pack(pady=(0, 4))

    tk.Button(win, text="閉じる",
              bg="#313244", fg=FG, font=("Yu Gothic UI", 9),
              relief="flat", padx=16, pady=6, cursor="hand2",
              command=win.destroy).pack(pady=(0, 16))


# ===== GUI =====
BG, BG2, BG3 = "#1a1a2e", "#16213e", "#0f3460"
FG, FG_DIM   = "#e0e0e0", "#888899"
ACCENT, GREEN, YELLOW, RED = "#4fc3f7", "#69f0ae", "#ffd54f", "#ef5350"


class StepCard(tk.Frame):
    def __init__(self, parent, num, title):
        super().__init__(parent, bg=BG2)
        hdr = tk.Frame(self, bg=BG3)
        hdr.pack(fill="x")
        tk.Label(hdr, text=f" STEP {num} ", bg=ACCENT, fg=BG,
                 font=("Yu Gothic UI", 9, "bold"), padx=6, pady=2).pack(side="left")
        tk.Label(hdr, text=f"  {title}", bg=BG3, fg=FG,
                 font=("Yu Gothic UI", 10, "bold"), pady=4).pack(side="left")
        self.body = tk.Frame(self, bg=BG2)
        self.body.pack(fill="x", padx=14, pady=10)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"文字起こしアプリ {APP_VERSION}")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.cfg   = load_config()
        self.log_q = queue.Queue()
        self._busy = False

        self._build()
        self._refresh_status()
        self._check_setup_banner()
        self.after(100, self._poll)

    def _build(self):
        hdr = tk.Frame(self, bg=BG3, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="文字起こしアプリ", bg=BG3, fg=ACCENT,
                 font=("Yu Gothic UI", 15, "bold")).pack()
        tk.Label(hdr, text="MP4を選んでボタンを押すだけ。発言者を自動で分けます。",
                 bg=BG3, fg=FG_DIM, font=("Yu Gothic UI", 9)).pack()

        wrap = tk.Frame(self, bg=BG, padx=16, pady=12)
        wrap.pack(fill="both")

        # ---- STEP 1: トークン ----
        s1 = StepCard(wrap, 1, "HuggingFaceのトークンを入力")
        s1.pack(fill="x", pady=(0, 8))

        tk.Label(s1.body,
                 text="初回のみ必要です。入力後は自動で記憶します。",
                 bg=BG2, fg=FG_DIM, font=("Yu Gothic UI", 9)).pack(anchor="w")

        row1 = tk.Frame(s1.body, bg=BG2)
        row1.pack(fill="x", pady=(6, 0))

        self.token_var = tk.StringVar(value=self.cfg.get("hf_token", ""))
        self.token_entry = tk.Entry(row1, textvariable=self.token_var,
                                    show="*", bg="#1e2a3a", fg=FG,
                                    insertbackground=FG, font=("Yu Gothic UI", 10),
                                    relief="flat", bd=6, width=34)
        self.token_entry.pack(side="left", fill="x", expand=True)

        self.show_var = tk.BooleanVar()
        tk.Checkbutton(row1, text="表示", variable=self.show_var,
                       command=lambda: self.token_entry.config(
                           show="" if self.show_var.get() else "*"),
                       bg=BG2, fg=FG_DIM, selectcolor=BG2,
                       activebackground=BG2, font=("Yu Gothic UI", 9)).pack(side="left", padx=4)

        tk.Button(row1, text="取得方法 ?",
                  bg="#1e3a4a", fg=ACCENT, relief="flat",
                  font=("Yu Gothic UI", 9), cursor="hand2", padx=8, pady=4,
                  command=lambda: show_token_guide(self)).pack(side="left", padx=(4, 0))

        # ---- STEP 2: ファイル ----
        s2 = StepCard(wrap, 2, "文字起こしするファイルを選ぶ")
        s2.pack(fill="x", pady=(0, 8))

        self.input_var = tk.StringVar()
        row2 = tk.Frame(s2.body, bg=BG2)
        row2.pack(fill="x")

        self.file_label = tk.Label(row2, text="ファイルが選ばれていません",
                                   bg="#1e2a3a", fg=FG_DIM, font=("Yu Gothic UI", 10),
                                   anchor="w", padx=8, pady=6, width=36)
        self.file_label.pack(side="left", fill="x", expand=True)

        tk.Button(row2, text="ファイルを選ぶ",
                  bg=ACCENT, fg=BG, font=("Yu Gothic UI", 10, "bold"),
                  relief="flat", cursor="hand2", padx=10, pady=4,
                  command=self._pick_file).pack(side="left", padx=(8, 0))

        # ---- STEP 3: 開始 ----
        s3 = StepCard(wrap, 3, "処理方式を選んで開始")
        s3.pack(fill="x", pady=(0, 8))

        device_row = tk.Frame(s3.body, bg=BG2)
        device_row.pack(fill="x", pady=(0, 8))

        tk.Label(device_row, text="処理方式:", bg=BG2, fg=FG,
                 font=("Yu Gothic UI", 10)).pack(side="left")

        self.device_var = tk.StringVar(value=self.cfg.get("device", "auto"))
        for val, label in [("auto", "自動"), ("cuda", "GPU優先"), ("cpu", "CPU")]:
            tk.Radiobutton(device_row, text=label, variable=self.device_var, value=val,
                          bg=BG2, fg=FG, selectcolor=BG3, activebackground=BG2,
                          font=("Yu Gothic UI", 10)).pack(side="left", padx=8)

        self.gpu_badge = tk.Label(device_row, text="", bg=BG2, fg=FG_DIM,
                                  font=("Yu Gothic UI", 9))
        self.gpu_badge.pack(side="left", padx=8)

        tk.Button(device_row, text="GPU有効化 →",
                  bg="#1a2a1a", fg=GREEN, relief="flat",
                  font=("Yu Gothic UI", 9), cursor="hand2", padx=8, pady=3,
                  command=lambda: show_gpu_guide(self)).pack(side="right")

        self.time_label = tk.Label(s3.body, text="",
                                   bg=BG2, fg=FG_DIM, font=("Yu Gothic UI", 9))
        self.time_label.pack(anchor="w", pady=(0, 8))

        self.start_btn = tk.Button(s3.body, text="▶  文字起こし開始",
                                   bg=GREEN, fg=BG, font=("Yu Gothic UI", 12, "bold"),
                                   relief="flat", padx=20, pady=8, cursor="hand2",
                                   command=self._start)
        self.start_btn.pack(anchor="w")

        self.progress = ttk.Progressbar(s3.body, mode="indeterminate", length=490)
        self.progress.pack(fill="x", pady=(10, 0))

        tk.Label(wrap, text="処理ログ", bg=BG, fg=FG_DIM,
                 font=("Yu Gothic UI", 9)).pack(anchor="w", pady=(8, 2))
        self.log_text = tk.Text(wrap, height=12, bg="#0d1117", fg=GREEN,
                                font=("Consolas", 9), relief="flat",
                                state="disabled", wrap="word", width=62)
        self.log_text.pack(fill="x")

        foot = tk.Frame(wrap, bg=BG)
        foot.pack(fill="x", pady=(8, 0))
        tk.Label(foot, text=f"文字起こしアプリ {APP_VERSION}  |  完全ローカル処理",
                 bg=BG, fg=FG_DIM, font=("Yu Gothic UI", 8)).pack(side="left")
        tk.Button(foot, text="保存フォルダを開く",
                  bg=BG2, fg=FG_DIM, font=("Yu Gothic UI", 9),
                  relief="flat", cursor="hand2",
                  command=self._open_folder).pack(side="right")

    def _check_setup_banner(self):
        ok, msg = check_setup()
        if not ok:
            banner = tk.Frame(self, bg="#3a1a0e", pady=6)
            banner.pack(fill="x", before=self.nametowidget(self.winfo_children()[1]))
            tk.Label(
                banner,
                text="⚠  セットアップ未完了  — フォルダ内の「★セットアップ.bat」をダブルクリックしてください",
                bg="#3a1a0e", fg="#ffd54f",
                font=("Yu Gothic UI", 9, "bold"),
                padx=12, pady=0
            ).pack(side="left")

    def _refresh_status(self):
        has_cuda, gpu_name, _ = detect_gpu()
        if has_cuda:
            self.gpu_badge.config(
                text=f"✓ {gpu_name}  使用可能",
                fg=GREEN, bg="#1a2a1a")
            self.time_label.config(
                text="処理時間の目安: 動画30分 → GPU約3〜5分")
        else:
            self.gpu_badge.config(
                text="GPU未対応（CPU版torch）",
                fg=YELLOW, bg="#2a2a0e")
            self.time_label.config(
                text="処理時間の目安: 動画30分 → CPU約60〜90分  ※「GPU有効化」で大幅短縮できます")

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="ファイルを選択",
            filetypes=[("動画・音声", "*.mp4 *.mov *.mkv *.avi *.m4a *.wav *.mp3"),
                       ("すべて", "*.*")]
        )
        if path:
            self.input_var.set(path)
            self.file_label.config(text=Path(path).name, fg=FG)

    def _open_folder(self):
        folder = self.cfg.get("output_dir", str(OUTPUT_DIR))
        Path(folder).mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", folder])

    def _log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _poll(self):
        try:
            while True:
                kind, val = self.log_q.get_nowait()
                if kind == "log":
                    self._log(val)
                elif kind == "done":
                    self._on_done(val)
                elif kind == "error":
                    self._on_error(val)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _on_done(self, path):
        self.progress.stop()
        self.start_btn.config(state="normal", bg=GREEN)
        self._busy = False
        self.cfg["hf_token"] = self.token_var.get()
        self.cfg["device"]   = self.device_var.get()
        save_config(self.cfg)
        messagebox.showinfo("完了！",
            "文字起こしが完了しました！\n\n"
            "「保存フォルダを開く」ボタンで結果を確認できます。")

    def _on_error(self, msg):
        self.progress.stop()
        self.start_btn.config(state="normal", bg=GREEN)
        self._busy = False
        self._log(f"\n❌ エラー:\n{msg}")
        short = msg.split("\n")[0]
        messagebox.showerror("エラーが発生しました", f"{short}\n\n詳細はログを確認してください。")

    def _start(self):
        if self._busy:
            return

        token  = self.token_var.get().strip()
        infile = self.input_var.get().strip()
        outdir = self.cfg.get("output_dir", str(OUTPUT_DIR))
        model  = self.cfg.get("model", "large-v3")

        if not token or not token.startswith("hf_"):
            messagebox.showwarning("STEP 1 が未完了",
                "HuggingFaceのトークンを入力してください。\n\n"
                "「取得方法 ?」ボタンで手順を確認できます。")
            return
        if not infile:
            messagebox.showwarning("STEP 2 が未完了",
                "「ファイルを選ぶ」ボタンでファイルを選んでください。")
            return

        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

        self._busy = True
        self.start_btn.config(state="disabled", bg="#445544")
        self.progress.start(12)
        self.cfg["hf_token"] = token
        self.cfg["device"]   = self.device_var.get()
        save_config(self.cfg)

        threading.Thread(
            target=run_transcription,
            args=(infile, outdir, token, model, self.device_var.get(), self.log_q),
            daemon=True
        ).start()


if __name__ == "__main__":
    App().mainloop()
