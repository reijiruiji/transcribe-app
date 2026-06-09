import re, pathlib, sys

files = {
    "QIITA.md":      r"C:\Users\syuri\Videos\画面録画\_app\QIITA.md",
    "NOTE_DRAFT.md": r"C:\Users\syuri\Videos\画面録画\_app\NOTE_DRAFT.md",
}

EXPECTED_PREFIXES = [
    "https://github.com/reijiruiji/",
    "https://qiita.com/reijiruiji/",
    "https://huggingface.co",
    "https://pytorch.org/",
    "https://download.pytorch.org/",
    "https://github.com/m-bain/",
    "https://github.com/pyannote/",
    "https://github.com/openai/",
    "https://docs.python.org/",
    "https://ffmpeg.org/",
]

all_ok = True
for name, path in files.items():
    text = pathlib.Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    issues = []

    # 未閉じコードブロック
    fence_count = sum(1 for l in lines if l.strip().startswith("```"))
    if fence_count % 2 != 0:
        issues.append("コードブロックが閉じられていない可能性")

    # URL 抽出
    urls = re.findall(r"https?://[^\s\)\]\"'`]+", text)
    unknown = [u for u in urls if not any(u.startswith(p) for p in EXPECTED_PREFIXES)]
    if unknown:
        issues.append(f"未確認URL: {unknown}")

    # 古いアカウント名チェック
    if "handamin/transcribe-app" in text:
        issues.append("古いURL handamin/transcribe-app が残っている")

    status = "OK  " if not issues else "WARN"
    print(f"[{status}] {name}  ({len(text)}文字, {len(lines)}行)")
    for i in issues:
        print(f"       !! {i}")
        all_ok = False

print()
print("全チェック完了" if all_ok else "警告あり")
sys.exit(0 if all_ok else 1)
