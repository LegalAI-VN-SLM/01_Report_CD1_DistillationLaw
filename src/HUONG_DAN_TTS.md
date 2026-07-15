# Hướng dẫn Text-to-Speech (TTS) tiếng Việt cho bài thuyết trình

Đọc `presentation_script_v2.md` (và các file tóm tắt/QA) thành **giọng nói tiếng Việt**, phát âm **thuật ngữ tiếng Anh cho chuẩn**.

**2 engine:**
- **VieNeu-TTS** (`tts_vieneu.py`) — chạy **trên máy (offline)**, giọng **rất tự nhiên**, KHÔNG lỗi mạng. Chậm hơn trên CPU (~80s/slide) và tốn ~5.7GB tải model lần đầu. **Khuyên dùng khi mạng chập chờn / cần chất lượng.**
- **edge-tts** (`tts_md.py`, `tts_presentation.py`) — gọi server Microsoft, **nhanh khi mạng ổn** nhưng hay lỗi `NoAudioReceived` lúc server bận. Miễn phí, không cần model.

---

## 0. VieNeu — chạy offline, giọng tự nhiên (khuyên dùng)

```powershell
# Cài (một lần); lần chạy đầu tự tải model ~5.7GB vào cache HF:
python -m pip install vieneu

cd src
python tts_vieneu.py --list-voices          # xem 10 giọng

# File MD (tóm tắt / QA) -> 1 wav:
python tts_vieneu.py --input ..\4_docs\TOM_TAT_BAO_CAO_3trang.md --output ..\4_docs\audio\tom_tat.wav
python tts_vieneu.py --input ..\4_docs\QA_VAN_DAP_THEO_TUNG_PHAN.md --output ..\4_docs\audio\qa.wav --voice "Ngọc Linh"

# Bài thuyết trình (mỗi slide 1 wav):
python tts_vieneu.py --input ..\4_docs\02_presentation\presentation_script_v2.md --outdir ..\4_docs\02_presentation\audio --range 1-19
# hoặc gộp cả bài (--combined), hoặc cả 28 slide (bỏ --range).
```
- **Giọng** (`--voice`): Bình An (mặc định, nam điềm đạm), Ngọc Lan, Gia Bảo, Thái Sơn, Đức Trí, Mỹ Duyên, Trúc Ly, Xuân Vĩnh, Trọng Hữu, Ngọc Linh.
- Xuất **.wav** 48kHz. `--debug` để xem trước text (không cần nạp model). Đoạn lỗi tự bỏ qua.
- Có **GPU** thì VieNeu tự chuyển sang PyTorch → nhanh hơn nhiều; CPU thì chậm nhưng ổn.

---

## 1. TL;DR — cài gì, chạy gì

```bash
# 1) Cài (một lần) — dùng Python của venv report:
"E:\DoCode\1 VN-Legal-AI\01_Report_CD1_DistillationLaw\.venv\Scripts\python.exe" -m pip install edge-tts

# 2) Vào thư mục src:
cd "E:\DoCode\1 VN-Legal-AI\01_Report_CD1_DistillationLaw\src"

# 3a) Cả bài chính (slide 1–19) → 1 file mp3:
..\.venv\Scripts\python.exe tts_presentation.py ^
    --input ..\4_docs\02_presentation\presentation_script_v2.md ^
    --outdir ..\4_docs\02_presentation\audio --range 1-19 --combined

# 3b) HOẶC mỗi slide 1 file (slide_04.mp3, ...) — tiện lồng vào PowerPoint:
..\.venv\Scripts\python.exe tts_presentation.py ^
    --input ..\4_docs\02_presentation\presentation_script_v2.md ^
    --outdir ..\4_docs\02_presentation\audio
```

> **edge-tts đã được cài sẵn** vào `.venv` của report (phiên bản 7.2.8). Nếu dùng máy/venv khác thì chạy lại lệnh `pip install edge-tts`.

---

## 2. Các script trong `src/`

| Script | Dùng cho | Đặc điểm |
|---|---|---|
| **`tts_presentation.py`** | **Bài thuyết trình** (`presentation_script_v2.md`) | Chỉ trích **lời thoại thật** (dòng 🎤 + câu *→ chuyển ý*), bỏ tiêu đề/mốc thời gian/emoji/mục "Mẹo". Sinh audio **theo từng slide** hoặc gộp cả bài. Có lớp phát âm `PRON_OVERRIDES`. |
| **`tts_md.py`** | **File markdown dài** (tóm tắt, QA…) — *khuyên dùng* | Đọc **toàn bộ** file (tự xử lý bảng/LaTeX/markdown/thuật ngữ, **không cần chỉnh tay**), **chia đoạn (chunk) → ghép** thành 1 mp3, có retry + bỏ qua đoạn lỗi. Dùng chung phiên âm với tts_presentation. |
| `tts_generator.py` | file md ngắn | Bản gốc: đọc cả file trong **1 lần gọi** edge-tts → **dễ lỗi `NoAudioReceived` với file dài**. Với file dài hãy dùng `tts_md.py`. |

Voice 2 file MD (KHÔNG cần sửa file md):
```bash
cd src
# Tóm tắt báo cáo:
..\.venv\Scripts\python.exe tts_md.py --input ..\4_docs\TOM_TAT_BAO_CAO_3trang.md ^
    --output ..\4_docs\audio\tom_tat.mp3
# Danh sách QA (giọng nữ):
..\.venv\Scripts\python.exe tts_md.py --input ..\4_docs\QA_VAN_DAP_THEO_TUNG_PHAN.md ^
    --output ..\4_docs\audio\qa.mp3 --voice vi-VN-HoaiMyNeural
# Xem trước text + số đoạn, không sinh audio:
..\.venv\Scripts\python.exe tts_md.py --input ..\4_docs\TOM_TAT_BAO_CAO_3trang.md --output x.mp3 --debug
```

> ⚠️ **Vì sao chia đoạn?** edge-tts hay trả `NoAudioReceived` khi **1 lần gọi quá dài** (đây là lỗi bạn gặp với `--combined`). `tts_md.py` và `--combined` của `tts_presentation.py` đã tự chia đoạn ~1800 ký tự, retry 5 lần, đoạn nào vẫn lỗi thì **bỏ qua + cảnh báo** (không làm hỏng cả file). edge-tts phụ thuộc server Microsoft nên đôi lúc chậm/chập chờn — nếu bị bỏ đoạn hoặc chạy chậm, **chạy lại** thường là ổn; cần chắc chắn hơn thì dùng backend ở mục 6.

---

## 3. Tham số hay dùng

| Tham số | Ý nghĩa | Ví dụ |
|---|---|---|
| `--input` | file md nguồn | `...presentation_script_v2.md` |
| `--outdir` | thư mục xuất mp3 (tts_presentation) | `..\4_docs\02_presentation\audio` |
| `--voice` | giọng | `vi-VN-NamMinhNeural` (nam), `vi-VN-HoaiMyNeural` (nữ) |
| `--rate` | tốc độ | `+0%` mặc định, `+8%` nhanh hơn, `-5%` chậm lại |
| `--range` | giới hạn slide | `1-19` (chỉ bài chính), `4-5` (2 slide mới) |
| `--combined` | gộp 1 file thay vì mỗi slide 1 file | (cờ) |
| `--debug` | **in text sẽ đọc, KHÔNG sinh audio** — luôn dùng để soát trước | (cờ) |

> 💡 **Luôn chạy `--debug` trước** để đọc thử văn bản đã làm sạch/phiên âm, chỉnh cho đúng rồi mới sinh mp3 (đỡ tốn thời gian gọi mạng).

---

## 4. Phát âm thuật ngữ tiếng Anh "cho chuẩn"

Vấn đề: giọng vi-VN đọc thẳng "distillation", "DPO", "ROUGE-L", "syllogism"… sẽ **sai/khó nghe**. Giải pháp: **thay trước khi đọc** bằng dạng tiếng Việt hoặc phiên âm chữ-cái.

Có **2 tầng**, áp theo thứ tự:
1. **`PRON_OVERRIDES`** trong `tts_presentation.py` — ưu tiên cao, sửa các ca đặc biệt (cụm dài trước).
2. **`tech_mappings`** trong `tts_generator.py` — bảng chính, dùng chung.

Một số phiên âm đang dùng (sửa được tùy ý):

| Thuật ngữ | Đọc thành | Ghi chú |
|---|---|---|
| Knowledge Distillation | (giữ tiếng Việt: "chưng cất tri thức") | script đã viết sẵn tiếng Việt |
| Syllogism / syllogistic | **tam đoạn luận** | dùng thuật ngữ Việt, tự nhiên nhất |
| ROUGE-L | **Ru-giơ Eo** | ROUGE đọc kiểu Pháp/Anh |
| QLoRA / LoRA | **Kiu Lo-ra** / **Lo-ra** | Q = "kiu" |
| DPO | **Đi Pi Âu** | đọc tên chữ kiểu Anh (dân CS quen) |
| GRPO | **Gờ Rờ Pi Âu** | |
| Diagnosis-Driven DPO | **Đi Pi Âu dẫn dắt bằng chẩn đoán** | cụm dài → thay nguyên cụm |
| Offline Logit KD | **chưng cất logit offline** | |
| On-Policy KD | **chưng cất On Policy** | |
| SFT / NLI / MCQ / SLM / LLM | đọc từng chữ ("S F T"…) | |
| GPT-4o-mini | **Gi Pi Ti bốn ô mini** | |
| exposure bias | **thiên kiến phơi nhiễm** | |
| catastrophic forgetting | **quên đi thảm họa** | |

**Muốn thêm/sửa phát âm?** Mở `tts_presentation.py`, sửa dict `PRON_OVERRIDES` (đặt **cụm dài lên trên** để thay trước từ ngắn bên trong). Ví dụ thêm:
```python
"checkpoint": "chếch-point",
"pipeline": "pai-lai",
```

> Lưu ý: đây là **phiên âm gần đúng** cho giọng vi-VN. Nếu cần phát âm English *thật* chuẩn (giọng Anh xen giữa), xem mục 6 (giọng đa ngôn ngữ / viXTTS).

---

## 5. Lồng audio vào PowerPoint / dựng video

- Sinh **mỗi slide 1 file** (`slide_04.mp3`, `slide_05.mp3`, …) bằng cách bỏ `--combined`. Tên file khớp **số slide trong deck** → PowerPoint: *Insert → Audio → Audio on My PC*, đặt "Play automatically", một file cho mỗi slide.
- Hoặc dùng skill **ppt-master** (`~/.agents/skills/ppt-master`) có sẵn `notes_to_audio.py` + workflow `generate-audio` để **đọc speaker notes và nhúng narration thẳng vào PPTX** (xuất video). Phù hợp vì deck `VN_Legal_AI_CD1_v2.pptx` đã có notes đầy đủ 28 slide.

---

## 6. Nâng cao / giải pháp thay thế

| Cần | Dùng | Ghi chú |
|---|---|---|
| Miễn phí, nhanh, đủ tốt | **edge-tts** (đang dùng) | cần mạng; giọng vi-VN neural |
| Giọng rất tự nhiên / clone giọng | **ElevenLabs**, **MiniMax**, **Qwen-TTS** | tốn phí/API key; ppt-master hỗ trợ backend này |
| Chạy **offline** + đọc **English xen tiếng Việt** tốt | **viXTTS** (Coqui XTTS-v2 fine-tune tiếng Việt) hoặc **CosyVoice** | cài nặng hơn (torch, model vài GB), chạy GPU; đọc thuật ngữ Anh tự nhiên hơn edge-tts |
| Gộp nhiều mp3 thành 1 (nếu để riêng slide) | **ffmpeg** | `ffmpeg -f concat -i list.txt -c copy out.mp3`. Hiện máy **chưa có ffmpeg**; hoặc cứ dùng `--combined` để khỏi cần ffmpeg |

---

## 7. Lỗi thường gặp

- **`edge-tts is not installed`** → `python -m pip install edge-tts` (đúng venv).
- **`No audio was received`** (edge-tts) → text đầu vào rỗng hoặc chỉ có ký hiệu; kiểm bằng `--debug`, hoặc thu hẹp `--range`.
- **Treo / lỗi mạng** → edge-tts gọi server Microsoft, cần Internet; thử lại hoặc đổi mạng.
- **Đọc sai số/thuật ngữ** → chạy `--debug`, chỉnh `PRON_OVERRIDES` (tts_presentation) hoặc `tech_mappings` (tts_generator).
- **Chữ tiếng Việt trên console bị vỡ** → chỉ là hiển thị; đặt `PYTHONIOENCODING=utf-8`. File mp3 vẫn đúng.
