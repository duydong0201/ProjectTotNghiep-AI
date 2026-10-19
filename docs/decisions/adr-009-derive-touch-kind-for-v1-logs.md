# ADR-009: Suy ra `rally_last_touch_kind` cho log v1, thay vì chơi lại

- **Ngày:** 14/10/2026
- **Trạng thái:** Đã áp dụng
- **Liên quan:** ADR-005 (dựng lại frame cho log v0) — cùng một kiểu quyết định

## Bối cảnh

Schema v2 thêm cột `rally_last_touch_kind` (loại cú chạm gần nhất), cần cho 6 feature mô
tả chuỗi phối hợp. Cột này chỉ được game ghi **từ 12/10/2026**.

Nhưng 162 trận người chơi thật đã thu **trước** mốc đó:

| Tập | Số trận | Ngày |
|---|---|---|
| `human_real` | 12 | 09-10, 10-10 |
| `human` | 120 | 10-10 |
| `human_holdout` | 30 | 10-10, 10-11 |

Hai lựa chọn: chơi lại toàn bộ (tốn thời gian người chơi), hoặc suy ra cột đó từ dữ liệu
đã có.

## Quyết định

Suy ra cột, bằng `rebuild_kind_v1_to_v2()` trong `src/spike_ai/data.py`.

Thông tin cần thiết đã nằm trong log v1:

- `rally_last_touch` — ai vừa chạm bóng
- `rally_touch_count` — lần chạm thứ mấy
- `p_input_intent` / `o_input_intent` — nhân vật đó phát ra ý định gì

Thuật toán: khi phát hiện một cú chạm mới (`lastTouch` đổi **hoặc** `touchCount` đổi),
nhìn lui tối đa 12 frame trong cột intent của **chính** nhân vật đó để tìm intent chạm
bóng, rồi ánh xạ `FinalIntent → ActionState`.

Ba điều kiện biên đã làm sai lúc đầu, và từng cái đều đo được tác hại:

| Điều kiện | Hậu quả khi thiếu |
|---|---|
| `k == 0` (dòng đầu file đã có cú chạm sẵn) | mất cả pha bóng đầu — 73/5385 frame sai |
| `lastTouch == 8` là reset, **không phải** −1 (`GameConfig::NONE = 8`) | giữ cú cũ qua pha bóng mới |
| `touchCount` đổi cũng là cú chạm mới | bỏ sót khi cùng nhân vật chạm hai lần |
| `Jump` không chạm bóng (`JumpStartSystem`, không qua `UpdateRallyTouch`) | nhận sai `Jump` thành cú chạm |

## Kiểm chứng

Có 2 file log v2 thật chứa **cả** cột thực tế lẫn các cột dùng để suy ra, nên đo được
trực tiếp:

```
2026-10-12_17-45-24_BOTRULE.csv   100.00%  (3961 frame)
2026-10-14_16-34-53_BOTRULE.csv   100.00%  (5385 frame)
TỔNG: 9346/9346 frame khớp (100.00%)
```

`scripts/upgrade_v1_to_v2.py --verify <dir>` đo lại mỗi lần chạy, nên không phải tin vào
con số cũ. Thêm 6 unit test trong `tests/test_data.py` chốt từng điều kiện biên.

**Hạn chế của phép kiểm chứng:** hai file đối chiếu là bot-vs-bot nên **không có `Set`
và `Bump`** lần nào. Phần suy ra cho đúng hai loại cú đó chưa đối chiếu được với dữ liệu
thật, dù cơ chế là chung cho mọi loại cú. Muốn đóng hẳn lỗ này thì cần một trận **người
chơi** thu bằng bản build v2.

## Hệ quả về nguồn gốc dữ liệu

Cột này là **suy ra**, không phải do game ghi trực tiếp. Hai lớp bảo vệ để điều đó không
bị quên:

1. `upgrade_v1_to_v2.py` **từ chối** ghi nếu tên thư mục đích không chứa `derive`.
2. `load_match()` gán tên thư mục vào cột `source`, nên dấu nguồn gốc đi theo dữ liệu
   vào lúc train (`source = "human_real_kind_derived"`).

Khi báo cáo phải nói rõ: con số đo trên tập này dùng một cột suy ra với độ chính xác
100% đã kiểm trên 9.346 frame, không phải cột gốc.
