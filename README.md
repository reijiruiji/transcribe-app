---
title: 完全ローカル・発言者分離つき文字起こしアプリをPythonで作った【WhisperX + pyannote】
tags:
  - Python
  - 音声認識
  - WhisperX
  - 機械学習
  - tkinter
---

**このアプリは2ステップで使えます。**
---
**① `setup.bat` をダブルクリック（最初の1回だけ）**
DL後、黒い画面が出て必要なものを自動インストールします。完了するとデスクトップにショートカットが作られます。

**② デスクトップの「transcribe」をダブルクリック（毎回）**
アプリが開いたら、MP4を選んで「▶ 文字起こし開始」を押すだけです。

**それだけで、発言者ごとに分けた文字起こしファイルが保存されます。**

以降はこの記事を読まなくて大丈夫です。

---

## はじめに

会議や取材の録画を文字起こしするとき、こんな悩みありませんか？

- クラウドサービスに音声を上げるのが**セキュリティ的に怖い**
- 文字起こしはできても**誰が話したかわからない**
- エンジニアじゃない人でも使えるように**GUIにしたい**

この記事では、これを全部解決する**完全ローカル動作・発言者分離つきの文字起こしアプリ**をPythonで作った話を書きます。

完成したアプリはこちら：
👉 [GitHub - transcribe_app](https://github.com/reijiruiji/transcribe-app)

---

## 完成品

MP4を選んでボタンを押すだけ。

```
【SPEAKER_00】
  00:05.20 - 00:12.80  本日はお集まりいただきありがとうございます。

【SPEAKER_01】
  00:13.50 - 00:20.10  よろしくお願いします。早速ですが...

【SPEAKER_00】
  00:21.00 - 00:35.40  はい、まず最初の議題ですが...
```

「誰が話したか」を自動で識別して、タイムスタンプつきで出力します。

---

## 技術構成

| コンポーネント | 役割 |
|---|---|
| **WhisperX** | 文字起こし＋単語レベルのタイムスタンプ合わせ |
| **pyannote.audio** | 話者分離（誰がいつ話したかの識別） |
| **PyTorch (CUDA)** | GPU高速処理 |
| **ffmpeg** | 動画→音声変換 |
| **tkinter** | GUI（Python標準ライブラリ） |

### WhisperXを選んだ理由

OpenAIのWhisperをそのまま使うと「話者分離」ができません。  
WhisperXはWhisperに以下を追加したラッパーです：

1. **単語レベルのアライメント**（どの単語が何秒か正確に把握）
2. **pyannote.audioとの統合**（話者分離との組み合わせが簡単）

これにより「AさんがX秒〜Y秒に話した→テキストはこれ」という紐付けが可能になります。

---

## 環境

- Windows 11
- Python 3.12（3.14はWhisperXが非対応のため注意）
- NVIDIA RTX 4060 Ti（CPU動作も可）

:::note warn
**Python 3.14は非対応です。**
WhisperXが依存するctranslate2がPython 3.14に未対応のため、
必ず **Python 3.12** を使ってください。
:::

---

## セットアップ

### 1. Python 3.12 のインストール

```powershell
winget install Python.Python.3.12 --source winget
```

### 2. ffmpeg のインストール

```powershell
winget install Gyan.FFmpeg --source winget
```

### 3. whisperx のインストール

```powershell
py -3.12 -m pip install whisperx
```

### 4. GPU使用時（NVIDIA）

```powershell
# CUDA 12.8対応版（RTX 4060 Tiなどで動作確認済み）
py -3.12 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

CPUでも動きますが、処理時間が大幅に変わります：

| 動画30分 | CPU | GPU（RTX 4060 Ti） |
|---|---|---|
| 処理時間 | 約60〜90分 | 約3〜5分 |

### 5. HuggingFaceトークンの取得

発言者分離にはpyannote.audioのモデルを使います。利用に**HuggingFaceの無料アカウント**が必要です。

1. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) でトークンを作成
2. 権限は **この1つだけ** チェック：
   ```
   ✅ Read access to contents of all public gated repos you can access
   ```
3. 以下のモデルページで利用規約に同意（リンクを開いてAgreeを押すだけ）：
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

:::note
トークンは最小権限の「gated repoの読み取り」だけで十分です。
Write権限など不要なものは付けないようにしましょう。
:::

---

## コードの解説

### 文字起こし処理の流れ

```python
import whisperx
import gc

def run_transcription(input_file, hf_token, device="cuda"):
    compute_type = "float16" if device == "cuda" else "int8"

    # 1. 音声読み込み
    audio = whisperx.load_audio(input_file)

    # 2. 文字起こし（Whisper large-v3）
    model = whisperx.load_model("large-v3", device, compute_type=compute_type, language="ja")
    result = model.transcribe(audio, batch_size=16)
    del model; gc.collect()  # メモリ解放

    # 3. 単語レベルのタイムスタンプ合わせ
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device)
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False)
    del align_model; gc.collect()

    # 4. 話者分離
    diarize = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize(audio)

    # 5. 文字起こし結果に話者ラベルを割り当て
    result = whisperx.assign_word_speakers(diarize_segments, result)

    return result
```

ポイントは `del model; gc.collect()` でモデルを都度解放すること。  
large-v3モデルは約3GBのVRAMを使うため、次のモデルをロードする前に解放が必要です。

### GPU自動検出

```python
def detect_device(preferred: str) -> str:
    """'auto'|'cuda'|'cpu' → 実際のデバイス文字列"""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if preferred == "auto":
        return "cuda" if has_cuda else "cpu"
    if preferred == "cuda" and not has_cuda:
        return "cpu"  # GPUを選んでも使えなければCPUにフォールバック
    return preferred
```

### 長時間処理をGUIがフリーズしないようにする

文字起こしは数十分かかることがあります。`threading` でバックグラウンド実行し、`queue` でUIに進捗を送ります。

```python
import threading
import queue

log_q = queue.Queue()

def run_in_background():
    # 処理中にキューにメッセージを送る
    log_q.put(("log", "【2/4】文字起こし中..."))
    # ... 処理 ...
    log_q.put(("done", output_path))

# バックグラウンドで実行
threading.Thread(target=run_in_background, daemon=True).start()

# GUIは100msごとにキューを確認してログを表示
def poll():
    try:
        kind, val = log_q.get_nowait()
        if kind == "log":
            log_text.insert("end", val + "\n")
        elif kind == "done":
            messagebox.showinfo("完了！", val)
    except queue.Empty:
        pass
    root.after(100, poll)  # 100ms後に再チェック
```

---

## GUI全体の設計

tkinterで「ステップ形式」のUIにしました。STEP番号が見えることで「今何をすればいいか」が迷わずわかります。

```
STEP 1: HuggingFaceトークンの入力
STEP 2: ファイルを選ぶ
STEP 3: 処理方式を選んで開始（GPU/CPU切り替え）
```

「取得方法 ?」ボタンを押すとポップアップでトークン取得手順を表示するようにして、アプリを閉じなくても手順を確認できます。

---

## ハマったポイント

### Python 3.14 非対応問題

最初に `py -m pip install whisperx` を実行したところ：

```
ERROR: Could not find a version that satisfies the requirement ctranslate2==4.4.0
```

原因は Python 3.14 でした。ctranslate2 が 3.14 未対応のため、`py -3.12` で明示的に 3.12 を指定する必要があります。

### バッチファイルの文字化け

`.bat` ファイルを UTF-8 で保存すると BOM が付いてしまい、cmd.exe が先頭バイトを誤読してコマンドが壊れます。

```
'scription' は、内部コマンドまたは外部コマンド...
```

解決策：PowerShellで Shift-JIS (CP932) として書き込む。

```powershell
[System.IO.File]::WriteAllText(
    'path\to\file.bat',
    $content,
    [System.Text.Encoding]::GetEncoding(932)
)
```

または bat ファイル内の日本語を全部取り除いてASCIIのみにする（より確実）。

### CPU版torchが入ってしまう問題

`pip install whisperx` で torch が自動インストールされますが、これは **CPU版** です。  
NVIDIA GPUを使うには別途 CUDA 対応版を上書きインストールする必要があります。

```powershell
# これを後から実行する
py -3.12 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

`pip install whisperx` だけでは GPU は使えないので注意。

---

## まとめ

| 機能 | 対応 |
|---|---|
| 完全ローカル処理 | ✅ |
| 発言者の自動識別 | ✅ |
| GPU高速処理 | ✅（RTX 4060 Tiで約10倍速） |
| GUIアプリ | ✅（tkinter） |
| 日本語音声 | ✅（whisper large-v3） |
| Python未導入者向けsetup.bat | ✅ |

プライバシーが気になる場面での文字起こしに使えます。  
「ファイルを選んで押すだけ」にしたことで、エンジニア以外の人にも渡せるのが気に入っています。

コードはGitHubに公開しています。

👉 [GitHub - transcribe-app](https://github.com/reijiruiji/transcribe-app)

---

## 参考リンク

- [WhisperX](https://github.com/m-bain/whisperX)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [Whisper (OpenAI)](https://github.com/openai/whisper)
- [PyTorch — CUDA Wheels](https://pytorch.org/get-started/locally/)
