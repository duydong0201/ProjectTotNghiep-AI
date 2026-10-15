# ADR-010: 6 feature chuỗi phối hợp (schema v2) không cải thiện điểm offline

- **Ngày:** 14/10/2026
- **Trạng thái:** Kết quả âm, ĐÃ CHỐT. Xác nhận 2 lần offline (12 và 315 trận) và 1 lần trong game.
- **Liên quan:** ADR-007 (feature vận tốc cũng là kết quả âm), ADR-009 (cách lấy dữ liệu v2)

## Bối cảnh

Đo ngày 12/10/2026: chuỗi `Set → Jump → Spike` xuất hiện **0 lần** trong game ở cả 3 lần
chạy, `Jump` chỉ 1 lần mỗi lần chạy.

Chẩn đoán lúc đó: model là **stateless** — mỗi frame chỉ thấy 21 con số của frame đó, nên
không có cách nào biết "mình vừa Set xong". Schema v1 chỉ có `last_touch_me` (ai chạm) và
`touch_count` (mấy lần), **không có loại cú**.

Giả thuyết: thêm feature mô tả loại cú chạm trước đó sẽ cho model học được chuỗi.

## Thí nghiệm

Schema v2 = v1 + cột `rally_last_touch_kind` → 6 feature:
`my_last_set`, `my_last_bump`, `my_last_spike`, `my_last_serve`, `opp_last_spike`,
`self_airborne`. 21 → 27 feature.

Phép so sánh **đổi đúng một biến**: cùng 12 trận người chơi thật, cùng split (salt không
đổi), cùng seed 42, cùng cửa sổ ±20 frame, cùng tham số Random Forest. Chỉ khác số feature.

(Rút kinh nghiệm: phép so sánh v0 → v1 trước đây đổi 4 biến cùng lúc nên không kết luận
được gì về nguyên nhân.)

## Kết quả

| | v1 (21 feature) | v2 (27 feature) |
|---|---|---|
| `action` macro-F1 | 0,352 ± 0,024 | **0,335 ± 0,022** |
| `move` macro-F1 | 0,637 | 0,634 |

Theo từng lớp:

| Lớp | n | v1 | v2 | Thay đổi |
|---|---|---|---|---|
| **`Jump`** | 155 | 0,062 | **0,062** | **+0,001** |
| `Set` | 174 | 0,435 | 0,433 | −0,002 |
| `Serve` | 45 | 0,915 | 0,819 | **−0,096** |
| `SpikeMedium` | 129 | 0,254 | 0,221 | −0,033 |
| `SpikeStrong` | 105 | 0,125 | 0,122 | −0,002 |
| `SpikeLight` | 34 | 0,048 | 0,050 | +0,003 |
| `None` | 20.734 | 0,794 | 0,788 | −0,007 |

`Jump` — chính lớp mà 6 feature nhắm vào — **không nhúc nhích**.

Model **có** dùng các feature mới (tổng độ quan trọng 12,62%; `my_last_serve` 5,8%,
`self_airborne` 4,4%), nhưng không học thêm được gì. Giải thích: 21.383 frame với 174 mẫu
`Set` và 155 mẫu `Jump` là chế độ ít dữ liệu — thêm 6 feature vào 21 làm tăng phương sai
mà không thêm tín hiệu, nên các lớp ít mẫu (`Serve` 45 mẫu) tụt nhiều nhất.

## Xác nhận lần hai, trên dữ liệu lớn hơn 25 lần (14/10/2026)

Chạy lại cùng phép so sánh trên **315 trận** người chơi (`Logs/v1`), thay vì 12 trận:

| | v1 (21 feature) | v2 (27 feature) |
|---|---|---|
| `action` random_forest | **0,671 ± 0,049** | 0,623 ± 0,030 |
| `action` decision_tree | **0,672 ± 0,036** | 0,589 ± 0,035 |
| `move` random_forest | 0,758 | 0,758 |

v2 **vẫn tệ hơn**, và khoảng cách còn rộng hơn lần đầu.

**Giải thích ban đầu của tôi là sai.** Lần đầu tôi cho rằng nguyên nhân là chế độ ít dữ
liệu: "21.383 frame với 174 mẫu `Set` thì thêm 6 feature làm tăng phương sai mà không
thêm tín hiệu". Nhưng lần này có **1.081 mẫu `Set`** (gấp 6 lần) và **272.206 frame**
(gấp 13 lần) mà kết quả không đổi chiều. Vậy 6 feature đó thêm **nhiễu**, không phải
thiếu dữ liệu để khai thác chúng.

Kết luận: **dùng schema v1**. Hạ tầng v2 vẫn giữ (xem mục cuối), nhưng không dùng để train.

## Vì sao CHƯA bỏ

Trong dự án này, điểm offline đã **3 lần** không dự báo được hành vi trong game:

| | offline | trong game |
|---|---|---|
| Model 150 trận | 0,892 | `None` ở 8.014/8.014 frame — bất động |
| Model 6 trận | 0,356 | hành động 562 lần |
| Model 12 trận | per-class tốt hơn mọi mặt | `argmax` ra cú đập ở 1,1% frame (bản 6 trận: 19,4%) |

Nên điểm offline bằng nhau **không** kết luận được là v2 vô dụng trong game. Phép đo quyết
định là: export model v2, đổi `AIInputSystem` sang `MlFeaturesV2`, rồi đếm số lần chuỗi
`Set → Jump → Spike` xuất hiện.

Tuy vậy, chứng cứ hiện có là **không hứa hẹn**, vì một lý do cụ thể: `Jump` F1 = 0,062 ở
cả hai bản, tức model vẫn không biết *khi nào* nên nhảy. Và chuỗi còn cần model đề xuất
`Set` trước, mà xác suất `Set` trong game đo được chỉ **0,025** — không feature nào ở đây
làm thay đổi con số đó.

## Việc giữ lại dù kết quả âm

Hạ tầng v2 vẫn giữ: cột trong game, spec, `_build_v2`, `MlFeaturesV2`, script nâng cấp.
Lý do: nó đúng và đã kiểm chứng (27/27 feature khớp C++ ↔ Python; 100% độ chính xác khi
suy ra cột), và nếu sau này thu thêm dữ liệu người chơi ở đúng vùng trạng thái thì giả
thuyết này đáng thử lại với nhiều mẫu hơn. Dùng hay không dùng chỉ là chọn config v1 hay v2.

## Điều cần nói trong báo cáo

Nút thắt thật **không phải** thiếu feature. Nó là **phân bố trạng thái**: người chơi `Set`
ở `touch_count ≈ 1,93` còn bot trong game gặp `touch_count ≈ 0,95`. Model không bao giờ ở
trong tình huống mà người chơi đã `Set`, nên dù có feature mô tả chuỗi thì cũng không có
dịp dùng. Cách chữa là thu dữ liệu ở đúng vùng trạng thái đó (DAgger), không phải thêm
feature nữa.
