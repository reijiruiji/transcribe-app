"""
ローカル文字起こし + 話者分離スクリプト
使用: python transcribe.py [動画ファイルパス]
"""

import sys
import gc
import whisperx
from pathlib import Path
from datetime import datetime

# ============================================================
# 設定
# ============================================================
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # HuggingFaceトークンをここに入力

OUTPUT_DIR = r"C:\Users\syuri\Videos\画面録画\文字起こし"
LANGUAGE = "ja"
MODEL_SIZE = "large-v3"   # 精度重視。遅い場合は "medium" や "small" に変更
DEVICE = "cpu"            # GPUがあれば "cuda"
COMPUTE_TYPE = "int8"     # CPU用。CUDA使用時は "float16"
BATCH_SIZE = 4            # CPU用。CUDA使用時は 8〜16
# ============================================================


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def transcribe(input_file: str) -> str:
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {input_file}")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 音声読み込み中: {input_path.name}")
    audio = whisperx.load_audio(str(input_path))

    print(f"[2/4] 文字起こし中 (モデル: {MODEL_SIZE}) ...")
    model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE)
    result = model.transcribe(audio, batch_size=BATCH_SIZE)
    del model
    gc.collect()

    print("[3/4] 単語レベル時刻合わせ中 ...")
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=DEVICE
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, DEVICE,
        return_char_alignments=False
    )
    del align_model
    gc.collect()

    print("[4/4] 話者分離中 ...")
    diarize_pipeline = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
    diarize_segments = diarize_pipeline(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # 出力ファイル名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_txt = output_path / f"{input_path.stem}_{timestamp}.txt"
    out_json = output_path / f"{input_path.stem}_{timestamp}.json"

    # テキスト出力（読みやすい形式）
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"元ファイル : {input_path}\n")
        f.write(f"作成日時   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        current_speaker = None
        for seg in result["segments"]:
            speaker = seg.get("speaker", "SPEAKER_??")
            start = seg["start"]
            end = seg["end"]
            text = seg["text"].strip()

            if not text:
                continue

            if speaker != current_speaker:
                f.write(f"\n【{speaker}】\n")
                current_speaker = speaker

            f.write(f"  {format_time(start)} - {format_time(end)}  {text}\n")

    # JSON出力（再利用・編集用）
    import json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=2)

    print(f"\n完了!")
    print(f"  テキスト : {out_txt}")
    print(f"  JSON     : {out_json}")
    return str(out_txt)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\syuri\Videos\画面録画\サイエンスアーツ.mp4"
    transcribe(target)
