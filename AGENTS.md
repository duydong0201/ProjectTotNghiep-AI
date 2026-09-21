# AGENTS.md

Hướng dẫn cho **AI trợ lý viết code** (Claude Code, Codex, Cursor, Copilot…) khi làm việc trong repo này.
Quy trình đầy đủ nằm ở [`docs/workflow.md`](docs/workflow.md). File này chỉ nhắc những luật dễ vi phạm nhất.

## Luật bắt buộc

1. **Đọc [`docs/workflow.md`](docs/workflow.md) trước lần sửa file đầu tiên.** Làm theo vòng A (code) hoặc vòng B (thí nghiệm).
2. **Không commit thẳng vào `main` hoặc `develop`.** Tạo nhánh `<type>/<slug>` từ `develop`.
3. **Schema là nguồn sự thật.** Đổi feature thì sửa `schema/*.json` trước, sau đó mới sửa `src/spike_ai/features.py`.
   Mọi feature phải tính lại được y hệt trong C++, nên chỉ dùng cộng/trừ, so sánh, đổi dấu.
4. **Không chạm vào holdout.** Không chạy `train --final`, không đổi `split.salt`, không đọc kết quả holdout
   để sửa feature/model, trừ khi người dùng yêu cầu rõ ràng.
5. **Chia dữ liệu theo nhóm (trận/người chơi)**, không bao giờ theo frame.
6. **Chạy `scripts/check.ps1` trước khi báo xong.** Không xoá, skip hay nới lỏng test để cho qua.
   Sửa bug thì thêm test tái hiện bug.
7. **Không commit** `data/`, `models/`, `exports/`, `.venv/`, file `.env` hay secret.
8. **Không sửa tay** file trong `exports/`. Chúng do `spike_ai.export_cpp` sinh ra.
9. Tài liệu viết bằng tiếng Việt. Cập nhật tài liệu có sẵn thay vì tạo file mới trùng nội dung.

## Bản đồ nhanh

| Cần tìm | Ở đâu |
|---|---|
| Kế hoạch theo giai đoạn | `PLAN.md` |
| Hợp đồng dữ liệu với repo game | `docs/data_contract.md`, `schema/` |
| Cách tích hợp vào C++ | `docs/integration.md` |
| Lý do các quyết định kỹ thuật | `docs/decisions/` |
| Kết quả thí nghiệm | `reports/experiments.md` |
