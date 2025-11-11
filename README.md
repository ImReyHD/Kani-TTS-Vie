# 😻 Kani TTS Vie

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/pnnbao97/Kani-TTS-Vie)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/pnnbao-ump/kani-tts-370m-vie)

<img width="500" height="500" alt="s83RYjg6BCrSPTNNXMu4u" src="https://github.com/user-attachments/assets/11384214-379c-4872-b464-c417d3b57458" />

Mô hình chuyển văn bản thành giọng nói tiếng Việt nhanh, rõ ràng và giàu biểu cảm, xây dựng trên nền tảng **Kani 370M**.
Kho lưu trữ này hỗ trợ cả **script chạy cục bộ** và các **demo UI/API** đi kèm với bản phát hành [pnnbao-ump/kani-tts-370m-vie](https://huggingface.co/pnnbao-ump/kani-tts-370m-vie) trên Hugging Face.

## Điểm nổi bật

* 🚀 **Inference nhanh** – khoảng 3 giây cho đoạn văn ngắn trên GPU đơn, hệ số thời gian thực ~0.1–0.3×.
* 🎭 **Đa giọng** – 18 giọng đọc, bao gồm Tiếng Việt, Tiếng Anh, Hàn, Đức, Tây Ban Nha, Trung và Ả Rập.
* 📓 **Notebooks đi kèm** – Hướng dẫn chi tiết inference, chuẩn bị dataset, và fine-tuning LoRA trong thư mục `finetune/`.

## Giọng đọc hỗ trợ

| Ngôn ngữ          | Giọng đọc                                                                          |
| ----------------- | ---------------------------------------------------------------------------------- |
| Tiếng Việt        | Khoa (Nam Bắc), Hùng (Nam Nam), Trinh (Nữ Nam)                                     |
| Tiếng Anh         | David (British), Puck (Gemini), Kore (Gemini), Andrew, Jenny (Irish), Simon, Katie |
| Tiếng Hàn         | Seulgi                                                                             |
| Tiếng Đức         | Bert, Thorsten (Hessisch)                                                          |
| Tiếng Tây Ban Nha | Maria                                                                              |
| Tiếng Trung       | Mei (Cantonese), Ming (Shanghai)                                                   |
| Tiếng Ả Rập       | Karim, Nur                                                                         |
| Trung lập         | Không có ID giọng (`None`)                                                         |


## Cấu trúc kho lưu trữ

* `main.py` – script CLI đơn giản (chạy batch).
* `gradio_app.py` – demo Gradio Blocks với loader động + nhiều giọng.
* `server.py` – dịch vụ FastAPI với các endpoint `/tts` và `/stream-tts`.
* `client/index.html` – frontend tĩnh giao tiếp với server FastAPI.
* `kani_vie/` – quản lý mô hình, helper streaming, và utilities cho audio player.
* `finetune/` – notebooks fine-tuning LoRA và chuẩn bị dataset.
* `requirements.txt` / `pyproject.toml` – manifest dependency (pip hoặc uv).

## Yêu cầu cài đặt

1. **Python 3.12** (hoặc phiên bản trong `.python-version`).
2. **Driver GPU + CUDA** tương thích với PyTorch.
3. **ffmpeg** (tùy chọn nhưng khuyến nghị cho xử lý audio).
4. **Token Hugging Face** có quyền truy cập checkpoints.

Cài đặt dependencies:

```bash
# Dùng uv (khuyến nghị)
uv sync

# Hoặc dùng pip
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cách sử dụng

### 1. Inference qua CLI

```bash
uv run python main.py \
  --text "Xin chào! Tôi là Kani TTS." \
  --speaker_id "nam-mien-nam"
```

Kết quả sẽ được lưu thành file WAV.

### 2. FastAPI + frontend tĩnh

Chạy API:

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

Phục vụ frontend tĩnh:

```bash
python -m http.server 3000 --directory client
```

Client hỗ trợ `/tts` (batch) và `/stream-tts` (PCM chunked) thông qua FastAPI.

### 4. Notebooks

* `kani-tts-inference.ipynb` – walkthrough chi tiết token layout, sampling, trộn giọng.
* `prepare_dataset.ipynb` – dọn dữ liệu, chuẩn hóa số, xây dựng shards.
* `finetune/kani-tts-vi-finetune.ipynb` – công thức fine-tuning LoRA.

Mở chúng bằng môi trường Jupyter sau khi kích hoạt virtual environment.

## Đóng góp

Chào đón mọi đóng góp!

1. Fork repository.
2. Tạo branch mới cho tính năng.
3. Chạy lint/tests liên quan.
4. Mở pull request mô tả cải tiến.

## Giấy phép

Dự án này phát hành theo [Apache License 2.0](LICENSE).
Kiểm tra giấy phép các mô hình/dataset bên thứ ba trước khi phân phối lại.



