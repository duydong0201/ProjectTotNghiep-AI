# ADR-014: Vòng tự học tự động - bot ML học từ những pha chính nó thắng (self-imitation)

- **Ngày:** 20/10/2026
- **Trạng thái:** Accepted
- **Liên quan:** ADR-011 (ghép nhãn), ADR-013 (đo theo policy của game)

## Bối cảnh

Người dùng muốn bỏ hẳn khâu người ngồi chơi: bot tự chơi, tự sinh tình huống, tự train, tự động hoàn
toàn, nhưng game chính (người vs bot) vẫn giữ nguyên. Có hai lựa chọn cho câu hỏi "khi không có
người, ai chấm nhãn":

- **A. Self-imitation:** bot ML chơi (có rút thăm nên mỗi pha một khác), chỉ giữ những pha bóng nó
  **thắng điểm** làm dữ liệu train.
- **B. Quy tắc ghi điểm viết tay làm chuyên gia DAgger.**

Người dùng chọn **A** (20/10/2026).

## Quyết định

**Phía game** (repo `ProjectTotNghiep`, nhánh `feat/match-runner`):

- `RunMode`: chế độ chạy đọc lúc chạy; mặc định = hằng số cũ, nên `MyGame.exe` vẫn là người vs bot.
- `MatchRunner.exe`: chạy bot luật vs bot ML **không cửa sổ** (~1.000× thời gian thật), vòng lặp chép
  đúng `MainScene::update()`, ghi log như game với `o_player_id = MLAUTO`. Dùng model riêng ở
  `Source/AI/GeneratedAuto/` nên không đụng model của game chính.
- Mọi nguồn ngẫu nhiên ở chế độ bot-vs-bot đi theo seed: cùng seed thì log giống hệt từng byte.

**Phía AI** (repo này):

- `train.build_rows` có mục `self_imitation`: với agent bot ML, giữ thêm frame của chính nó trong các
  trận tự sinh (`player_id = MLAUTO`), **chỉ ở pha bóng nó thắng** (`won_rally_mask`). Trọng số riêng,
  không nhân với `agent_weight`.
- `python -m spike_ai.selfplay --config configs/selfplay_v1.yaml`: mỗi vòng tự chơi N trận (log vào
  `Logs/auto/`, không trộn `Logs/v1`) → train trên `Logs/v1` + `Logs/auto` → export → build lại
  MatchRunner → đo bot ML vs bot luật trên **cùng bộ seed đo** cho mọi model → % điểm thắng cao hơn
  model đang giữ thì giữ, không thì trả lại model cũ.
- Model tốt nhất ở `models/<run>/best/`. Đưa vào game chính là việc làm tay, sau khi xem kết quả.

## Giới hạn (phải ghi trong báo cáo)

- Đây **không còn là học thuần từ người chơi**: phần dữ liệu tự sinh là bot học từ chính nó, với
  "phần thưởng" là thắng điểm. Dữ liệu người chơi (`Logs/v1`) vẫn nằm trong tập train.
- Đối thủ duy nhất là bot luật. Bot ML có thể học cách thắng **bot luật** chứ chưa chắc thắng người.
- Pha thắng có thể thắng nhờ bot luật đánh hỏng chứ không nhờ cú hay của bot ML; self-imitation vẫn
  chép cả pha đó. Chỉ số quyết định là % điểm thắng đo trên bộ seed cố định, không phải điểm offline.
- Nhãn là hành động của **cả con bot** (model + luật chặn + luật nhảy + phần di chuyển theo luật), nên
  model học bắt chước hành vi tổng hợp đó.
- Đo bằng 20 trận / model: độ nhiễu của % điểm thắng cỡ vài điểm phần trăm, nên chênh nhỏ chưa chắc
  là thật (`min_gain` để chặn).

## Lần chạy 1 (20/10/2026, chế độ `won_rallies`): không tiến bộ

50 trận tự chơi + 50 trận đo mỗi vòng, học MỌI frame của pha bóng bot ML thắng:

| Vòng | % điểm thắng | Set / trận | Jump / trận | Chuỗi Set→Jump→Spike | |
|---|---|---|---|---|---|
| xuất phát | 46,4% | 91,6 | 26,5 | 1.292 | |
| 1 | 45,8% | 99,4 | 27,8 | 1.292 | bỏ |
| 2 | 43,4% | 106,9 | 30,9 | 1.445 | bỏ |
| 3 | 46,0% | 114,1 | 31,3 | 1.459 | bỏ, tự dừng |

Bot **đổi cách chơi** (Set, Jump, chuỗi tăng dần) nhưng **không thắng nhiều hơn**. Lý do chính: học cả
những pha thắng nhờ bot luật đánh hỏng. Phụ: model mỗi vòng fit trên 90% dữ liệu, model xuất phát 100%.

## Sửa (20/10/2026)

- `self_imitation.mode: kills` - chỉ học **lượt cầm bóng cuối** của những pha bot ML **tự ghi điểm bằng
  cú đập** (`train.kill_possession_mask`). Còn ~7% số frame của pha thắng; nhãn chủ yếu Set / Jump /
  SpikeStrong. Trọng số 4 (~40k dòng / 2 vòng ≈ 160k, vẫn ít hơn 295k dòng người chơi).
- `split.dev_mode: none` - fit trên 100% dữ liệu, không đo offline, ngang hàng model xuất phát.

