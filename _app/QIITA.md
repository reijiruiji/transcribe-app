---
title: 【PC初心者OK】会議録画を「誰が何を話したか」ごと自動で文字起こしするアプリを作った（完全無料・ローカル動作）
tags:
  - Python
  - 音声認識
  - WhisperX
  - 機械学習
  - tkinter
---

## この記事でできるようになること

- MP4などの動画・音声ファイルを選ぶだけで**文字起こしが自動で完了**する
- **誰がいつ何を話したか**が自動で識別される（会議・インタビュー・対談などに便利）
- 音声データは**外部に一切送信されない**（セキュリティ的に安全）
- プログラミング不要。**ダブルクリックで起動**できる

**出力イメージ：**

```
【SPEAKER_00】
  00:05.20 - 00:12.80  本日はお集まりいただきありがとうございます。

【SPEAKER_01】
  00:13.50 - 00:20.10  よろしくお願いします。早速ですが...

【SPEAKER_00】
  00:21.00 - 00:35.40  はい、まず最初の議題ですが...
```

---

## パソコンに詳しくない方へ（エンジニア以外はここだけ読めばOK）

**2ステップで使えます。**

### ① 最初の1回だけ：`setup.bat` をダブルクリック

フォルダの中にある `★セットアップ（初回だけここをダブルクリック）.bat` をダブルクリックします。

```
黒い画面が開いて、必要なものが自動インストールされます。
5〜10分かかります。完了すると画面が閉じます。
デスクトップに「transcribe」というショートカットが作られます。
```

> **「何かインストールしていいですか？」と聞かれたら「はい」を選んでください。**

### ② 毎回の使い方：デスクトップの「transcribe」をダブルクリック

アプリが開いたら：

1. **STEP 1**：最初の1回だけ「トークン」を入力（→ 取得方法は後述）
2. **STEP 2**：「ファイルを選ぶ」で文字起こしたいMP4を選ぶ
3. **STEP 3**：「▶ 文字起こし開始」を押す

終わると「完了！」と表示されます。  
「保存フォルダを開く」ボタンで結果のテキストファイルを確認できます。

**それだけです。以降は②のダブルクリックだけで使えます。**

---

## 「トークン」って何？どこで取るの？

この「トークン」は、**誰が話したかを識別する機能を使うための鍵**のようなものです。

HuggingFaceという会社が提供しているAIモデルを使うために必要で、**無料で取得できます**。  
（クレジットカード不要・メールアドレスだけで登録できます）

### トークンの取得手順（5分でできます）

**1. アカウント作成**  
[https://huggingface.co](https://huggingface.co) にアクセスして「Sign Up」

**2. トークンを発行**  
ログイン後、右上のアイコン → Settings → Access Tokens → **「New token」**

**3. 権限は1つだけチェック**

```
✅ Read access to contents of all public gated repos you can access

（他は全部チェックなし）
```

> **なぜこの権限だけ？**  
> 「誰が話したか識別するAIモデル」は無断利用を防ぐため、HuggingFaceアカウントで利用同意が必要です。  
> この1つの権限さえあればそれを確認できます。  
> 書き込み権限などは一切不要です。

**4. 利用規約に同意（2つのモデル）**  
以下のページを開いて「Agree」を押すだけ（数秒で完了）：
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

**5. アプリのSTEP 1にトークン（`hf_` で始まる文字列）を貼り付け**  
次回から自動入力されます。

---

## GPUって何？使った方がいいの？

**GPU（グラフィックボード）とは**、ゲームや動画編集を高速化するパソコンの部品です。  
文字起こしの処理にも使えて、**あるとないとで処理速度が約10倍変わります。**

| あなたの状況 | おすすめ |
|---|---|
| NVIDIAのGPU（GeForce RTX / GTXなど）を積んでいる | GPU優先を選ぶ |
| ゲームをしない・普通のノートPC | CPU（遅いが動く） |
| よくわからない | 「自動」を選ぶ（アプリが判断してくれる） |

**処理時間の目安：**

| 動画の長さ | CPU | GPU（RTX 4060 Ti） |
|---|---|---|
| 10分 | 約20〜30分 | 約1〜2分 |
| 30分 | 約60〜90分 | 約3〜5分 |
| 60分 | 約2〜3時間 | 約6〜10分 |

> CPUでも**処理中はパソコンを使い続けられます**（バックグラウンドで動きます）。  
> 急ぎでなければCPUのままでも問題ありません。

---

## よくある質問

**Q. 処理中に「エラー」が出た**  
A. まず `setup.bat` を再度ダブルクリックしてみてください。途中で失敗していた可能性があります。

**Q. 黒い画面がすぐ閉じた**  
A. セットアップが正常に完了したサインです。デスクトップの「transcribe」ショートカットを探してください。

**Q. 文字起こし結果が途切れる・正確じゃない**  
A. 録音の音質が原因であることが多いです。雑音が少ない環境での録音ほど精度が上がります。  
また「large-v3」モデルを使用しているため、通常は高精度です。

**Q. 話者が「SPEAKER_00」「SPEAKER_01」のままで名前がわからない**  
A. アプリは自動で「人物を区別」はできますが「名前を識別」はできません。  
出力されたテキストファイルをメモ帳で開いて手動で書き換えてください。

**Q. MP4以外も使える？**  
A. MOV・MKV・AVI・MP3・WAV・M4Aも対応しています。

**Q. 音声データはクラウドに送られる？**  
A. 送られません。すべて自分のパソコンの中だけで処理します。

---

## 技術的な話（エンジニア向け）

### なぜWhisperXを選んだか

OpenAIのWhisperは単独では「話者分離」ができません。  
WhisperXはWhisperに以下を追加したラッパーです：

1. **単語レベルのアライメント**（各単語が何秒かを正確に把握）  
   → これがないと「誰の発言か」と「テキスト」を紐付けられない
2. **pyannote.audioとの統合**（話者分離との組み合わせが簡単）

「アライメント」とは「文字と音のタイミングを合わせる処理」のことです。

### 処理の流れ

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
    del model; gc.collect()  # VRAM解放（large-v3は約3GB使う）

    # 3. 単語レベルのタイムスタンプ合わせ（アライメント）
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

> **`del model; gc.collect()` について**  
> large-v3モデルはVRAM（GPU上のメモリ）を約3GB消費します。  
> 次のモデルをロードする前に明示的に解放しないとVRAM不足でクラッシュします。  
> （VRAMとは：GPU専用のメモリ。RAMとは別物）

### GUIがフリーズしない仕組み

文字起こしは数十分かかります。`threading` でバックグラウンド実行し、`queue` でUIに進捗を送ります。

```python
import threading
import queue

log_q = queue.Queue()

def run_in_background():
    log_q.put(("log", "【2/4】文字起こし中..."))
    # ... 処理 ...
    log_q.put(("done", output_path))

threading.Thread(target=run_in_background, daemon=True).start()

# GUIは100msごとにキューを確認
def poll():
    try:
        kind, val = log_q.get_nowait()
        if kind == "log":
            log_text.insert("end", val + "\n")
        elif kind == "done":
            messagebox.showinfo("完了！", val)
    except queue.Empty:
        pass
    root.after(100, poll)
```

### ハマったポイント

**Python 3.14 非対応問題**

```
ERROR: Could not find a version that satisfies the requirement ctranslate2==4.4.0
```

WhisperXが依存するctranslate2がPython 3.14に未対応。`py -3.12` で明示的に指定する必要があります。

**CPU版torchが入ってしまう問題**

`pip install whisperx` でtorchが自動インストールされますが、これは**CPU版**です。  
NVIDIA GPUを使うには別途CUDA対応版を上書きインストールする必要があります：

```powershell
py -3.12 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**バッチファイルの文字化け問題**

`.bat`をUTF-8で保存するとBOMが付いてcmd.exeが誤読します。  
解決策：bat内の日本語を取り除いてASCIIのみにするか、CP932で保存する。

---

## 技術構成まとめ

| コンポーネント | 役割 |
|---|---|
| [WhisperX](https://github.com/m-bain/whisperX) | 文字起こし＋単語レベルタイムスタンプ |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | 話者分離（誰がいつ話したか） |
| [ffmpeg](https://ffmpeg.org/) | 動画→音声変換 |
| [tkinter](https://docs.python.org/3/library/tkinter.html) | GUI（Python標準ライブラリ） |
| [PyTorch](https://pytorch.org/) | 機械学習バックエンド |

---

## まとめ

| 機能 | 対応 |
|---|---|
| 完全ローカル処理 | ✅ |
| 発言者の自動識別 | ✅ |
| GPU高速処理 | ✅（RTX 4060 Tiで約10倍速） |
| GUIアプリ | ✅（tkinter） |
| 日本語音声 | ✅（whisper large-v3） |
| 非エンジニア向けsetup.bat | ✅ |

プライバシーが気になる場面での文字起こしに使えます。  
「ファイルを選んで押すだけ」にしたことで、エンジニア以外の人にも渡せるのが気に入っています。

コードはGitHubに公開しています。

👉 **[GitHub - transcribe-app](https://github.com/reijiruiji/transcribe-app)**

---

## 参考リンク

- [WhisperX](https://github.com/m-bain/whisperX)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [Whisper (OpenAI)](https://github.com/openai/whisper)
- [PyTorch — CUDA Wheels](https://pytorch.org/get-started/locally/)
