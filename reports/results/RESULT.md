# Tình trạng hiện tại (23/10/2026)

## 2.1. Pipeline Behavior Cloning

Pipeline đầy đủ, từ dữ liệu trong game tới model chạy lại trong game, cộng thêm **vòng tự học (DAgger
tự động)**:

```text
Game (người chơi / bot tự chơi bằng MatchRunner)
  ↓
DatasetLogger: mỗi frame một dòng (schema v1) → Logs/v1, Logs/auto
  ↓
Preprocessing: ghép nhãn lùi 1 frame (label_shift), lọc người dạy, trọng số
  ↓
Training: Random Forest (100 cây, 21 feature)
  ↓
Evaluation: offline theo đúng policy của game + trận bot luật vs bot ML trên bộ seed mới
  ↓
Export model sang header C++ (golden test so khớp Python)
  ↓
AI sử dụng model trong game (model + luật chặn an toàn + vật lý chọn frame nhảy)
  ↺ vòng tự học: chuyên gia điều khiển một nửa số lượt → train lại → chỉ giữ model tốt hơn
```

## 2.2. Model đang dùng trong game

- Model: **`selfplay_expert_v10` vòng 2**, target `action` (None, Set, Bump, Jump, Serve, SpikeLight,
  SpikeMedium, SpikeStrong).
- 21 feature: vị trí và trạng thái hai nhân vật, vị trí, vận tốc và quỹ đạo bóng, khoảng cách tới lưới, số chạm.
- Dữ liệu train: log người chơi (30 trận thật + 298 trận tổng hợp) + frame của chuyên gia quy tắc từ vòng
  tự học (DAgger). Lượt chuyên gia ghi điểm và lượt đập ngay có trọng số ×6.
- Trong game: model chọn hành động và **quyết có nhảy** sau cú Set; vật lý chọn frame nhảy; luật chặn loại
  cú đập ra ngoài sân hoặc đập lưới.

## 2.3. Kết quả trong game (bot luật vs bot ML, 100 trận)

| Chỉ số | Bot luật | Bot ML |
|---|---:|---:|
| Trận thắng | 0 | **100** |
| % điểm thắng | 19,7% | **80,3%** |
| Điểm mất / trận | 13,8 | 3,4 |
| Set / Bump / Jump mỗi trận | 0 / 0 / 0 | 72 / 10 / 64 |
| Chuỗi Set → Jump → Spike | 0 | 6.317 |
| Đập ngay khi bóng vào vùng detect / trận | – | ~62 |

Bot ML ghi điểm bằng: SpikeStrong sau Set → Jump 55%, bot luật tự đánh hỏng 26%, giao bóng 9%.

## 2.4. Tiến triển qua các mốc (bộ trận để dành, seed 400000, 50 trận)

| Mốc | Cách nhảy sau Set | % điểm thắng |
|---|---|---:|
| Model học từ chuyên gia (v1) | rút thăm 31,4% | 48,3% |
| Xác nhận 2 bước (v3) | rút thăm 31,4% | 50,1% |
| + sửa luật chặn cú đập | rút thăm 31,4% | 72,6% |
| + model tự quyết nhảy | model quyết | **93,8%** |
| v10 vòng 2 (đang dùng, chơi đa dạng hơn) | model quyết | 82,0% |

## 2.5. Kết quả trước đây (lưu lại để so sánh)

Model `spike_choice` trên log v0 (6 feature, chưa có thông tin bóng): 101 trận, 241.542 frame, chỉ 41 mẫu
Spike sau khi lọc; 5-fold CV macro-F1 **0,485 ± 0,168** (SpikeLight 0,71 / SpikeMedium 0,71 / SpikeStrong
0,12). Hạn chế do thiếu cột bóng – đã giải quyết bằng schema v1.

Chi tiết từng thí nghiệm: [`reports/experiments.md`](../experiments.md); lý do các quyết định:
[`docs/decisions/`](../../docs/decisions/README.md).
