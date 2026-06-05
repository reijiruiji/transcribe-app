# 文字起こしアプリ

MP4などの動画・音声ファイルを選んでボタンを押すだけで、**発言者ごとに分けた文字起こし**ができるローカルアプリです。

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)

---

## パソコンに詳しくない方へ

**はじめての方は2ステップだけです。**

### ① 最初の1回だけ — `setup.bat` をダブルクリック

> 必要なプログラムを自動でインストールします。  
> 黒い画面が出て数分〜10分ほどかかります。完了すると自動で閉じます。  
> デスクトップに「transcribe」ショートカットが作られます。

### ② 毎回の使い方 — デスクトップの「transcribe」をダブルクリック

> アプリが開いたら：  
> 1. **STEP 1**：最初だけトークンを入力（「取得方法 ?」ボタンで手順が見られます）  
> 2. **STEP 2**：「ファイルを選ぶ」で文字起こししたいMP4を選ぶ  
> 3. **STEP 3**：「▶ 文字起こし開始」を押す  
>
> 終わると「完了！」と表示されます。「保存フォルダを開く」で結果を確認できます。

**それだけです。以降は②のダブルクリックだけで使えます。**

---

## 特徴

- **完全ローカル処理** — 音声データは外部に一切送信されません
- **発言者の自動分離** — 誰がいつ話したかを自動で識別
- **GPU対応** — NVIDIA GPUがあれば処理速度が約5〜10倍速くなります
- **シンプルなGUI** — ファイルを選んでボタンを押すだけ
- **自動保存** — トークン・設定は次回起動時に引き継がれます

---

## スクリーンショット

```
┌─────────────────────────────────────────────┐
│  文字起こしアプリ v2.0                        │
│  MP4を選んでボタンを押すだけ。発言者を自動で分けます │
├─────────────────────────────────────────────┤
│ STEP 1  HuggingFaceのトークンを入力           │
│  [hf_***********]  [表示] [取得方法 ?]        │
├─────────────────────────────────────────────┤
│ STEP 2  ファイルを選ぶ                        │
│  [会議録画.mp4        ]  [ファイルを選ぶ]      │
├─────────────────────────────────────────────┤
│ STEP 3  処理方式を選んで開始                  │
│  処理方式: ○自動  ●GPU優先  ○CPU            │
│  ✓ RTX 4060 Ti 使用可能                      │
│                                              │
│  [▶ 文字起こし開始]                           │
│  [========================進捗============]  │
├─────────────────────────────────────────────┤
│  処理ログ...                                  │
│  【1/4】音声を読み込んでいます...              │
│  【2/4】文字起こし中...                       │
└─────────────────────────────────────────────┘
```

---

## 出力例

```
元ファイル : 会議録画.mp4
作成日時   : 2026-06-03 14:32:10
============================================================

【SPEAKER_00】
  00:05.20 - 00:12.80  本日はお集まりいただきありがとうございます。

【SPEAKER_01】
  00:13.50 - 00:20.10  よろしくお願いします。早速ですが...
```

---

## セットアップ

### 動作環境

- Windows 10 / 11
- インターネット接続（初回セットアップのみ）
- NVIDIA GPU（任意・あると5〜10倍速い）

### インストール手順

**1. このリポジトリをダウンロード**

```
Code → Download ZIP → 解凍
```

**2. `setup.bat` をダブルクリック**

以下を自動でインストールします：
- Python 3.12（未インストールの場合）
- ffmpeg
- whisperx
- PyTorch（NVIDIA GPUがあれば自動でGPU版をインストール）

デスクトップにショートカットも自動作成されます。

**3. HuggingFaceトークンを取得**

発言者分離には [pyannote.audio](https://github.com/pyannote/pyannote-audio) を使用しています。  
初回のみ HuggingFace のトークンが必要です。

1. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) にアクセス（無料登録）
2. 「New token」をクリック
3. 権限は **1つだけ** チェック：
   ```
   ✅ Read access to contents of all public gated repos you can access
   ```
4. 作成されたトークン（`hf_` で始まる文字列）をアプリのSTEP 1に入力

> トークンはアプリ内に保存され、次回以降は自動入力されます。

**4. 以下のモデルの利用規約に同意**（HuggingFaceにログインした状態で）

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

---

## 使い方

1. デスクトップの「transcribe」ショートカットをダブルクリック
2. **STEP 1**：トークンを入力（初回のみ）
3. **STEP 2**：「ファイルを選ぶ」でMP4を選択
4. **STEP 3**：処理方式を選んで「▶ 文字起こし開始」を押す
5. 完了するとポップアップ通知 → 「保存フォルダを開く」で結果確認

---

## 処理時間の目安

| 動画の長さ | CPU | GPU（RTX 4060 Ti） |
|---|---|---|
| 10分 | 約20〜30分 | 約1〜2分 |
| 30分 | 約60〜90分 | 約3〜5分 |
| 60分 | 約2〜3時間 | 約6〜10分 |

---

## 技術構成

| コンポーネント | 用途 |
|---|---|
| [WhisperX](https://github.com/m-bain/whisperX) | 文字起こし＋単語レベルタイムスタンプ |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | 話者分離（誰が話したかの識別） |
| [ffmpeg](https://ffmpeg.org/) | 動画→音声変換 |
| [tkinter](https://docs.python.org/3/library/tkinter.html) | GUI |
| [PyTorch](https://pytorch.org/) | 機械学習バックエンド |

モデル: `openai/whisper-large-v3`（最高精度）

---

## ファイル構成

```
transcribe_app.py        # アプリ本体
setup.bat                # 初回セットアップ
transcribe_config.json   # 設定（自動生成）
文字起こし/               # 出力フォルダ（自動生成）
  ├── ファイル名_日時.txt  # 読みやすいテキスト
  └── ファイル名_日時.json # 詳細データ（タイムスタンプ付き）
```

---

## ライセンス

MIT License

---

## 参考・謝辞

- [WhisperX](https://github.com/m-bain/whisperX) by m-bain
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) by pyannote
- [Whisper](https://github.com/openai/whisper) by OpenAI
