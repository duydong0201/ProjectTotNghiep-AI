# Quy trình làm việc

Tài liệu này là **nguồn duy nhất** về cách làm việc trong repo. README chỉ link tới đây.
Quy trình được chắt lọc từ dự án nhóm P-037 và rút gọn cho repo một người làm.

## 1. Hai loại công việc, hai vòng lặp

Trong dự án ML có hai loại thay đổi khác nhau về bản chất:

| | Vòng A — Thay đổi code | Vòng B — Thí nghiệm |
|---|---|---|
| Ví dụ | thêm feature, sửa bug, đổi schema, thêm model mới vào `models.py` | đổi hyperparameter, thử bộ feature khác, so sánh model |
| Thứ thay đổi | `src/`, `tests/`, `schema/`, `docs/` | chỉ `configs/` và `reports/` |
| Branch | `feat/…`, `fix/…`, `docs/…`, `refactor/…`, `test/…`, `chore/…` | `exp/…` |
| Cần plan? | Có, nếu làm quá 1 buổi | Không, ghi thẳng vào `experiments.md` |

### Vòng A — Thay đổi code

```text
1. Tạo plan         docs/plans/active/<YYYY-MM-DD>-<slug>.md  (từ docs/plans/template.md)
2. Tạo branch       git switch develop && git pull && git switch -c feat/<slug>
3. Code + test      viết test cùng lúc với code; sửa bug => có test tái hiện bug
4. Chạy check       scripts/check.ps1                        (phải PASS trước khi commit)
5. Cập nhật docs    theo ma trận lan truyền ở mục 4
6. Commit           Conventional Commits (mục 3), mỗi commit là một thay đổi trọn vẹn
7. Push + PR        PR vào develop, chờ CI xanh, tự review diff rồi merge
8. Đóng việc        cập nhật WORKLOG.md, điền tổng kết plan, chuyển plan sang docs/plans/completed/
```

### Vòng B — Thí nghiệm

```text
1. Tạo config       configs/<ten-thi-nghiem>.yaml   (copy từ config gần nhất, chỉ đổi 1-2 thứ)
2. Tạo branch       git switch -c exp/<slug>  (từ develop)
3. Train            python -m spike_ai.train --config configs/<ten>.yaml
4. Đánh giá         chỉ nhìn kết quả dev / CV; đọc cảnh báo cân bằng; mở confusion matrix trong models/<run>/
5. Ghi nhận         điền cột "Ghi chú" trong reports/experiments.md: đổi gì, kết quả ra sao, vì sao
6. Quyết định
     - kết quả không đáng giữ  -> vẫn commit config + ghi chú (thí nghiệm thất bại cũng là dữ liệu)
     - kết quả đáng giữ        -> chốt vào reports/results/<YYYY-MM-DD>/ (mục 6)
     - quyết định kỹ thuật lớn -> viết ADR trong docs/decisions/
7. Commit + PR      PR vào develop
```

**Chế độ đánh giá** (`split.dev_mode` trong config, lý do ở [ADR-004](decisions/adr-004-data-split-strategy.md)):

| `dev_mode` | Khi nào dùng | Kết quả |
|---|---|---|
| `single` | nhiều trận (dữ liệu bot v1) | một số đo trên tập dev chia bằng hash |
| `kfold` | ít trận (v0, dữ liệu người) | trung bình ± độ lệch chuẩn của 5 fold theo trận |
| `lopo` | hỏi "có tổng quát sang người mới không?" | mỗi fold bỏ ra một người chơi |

So sánh hai model thì chênh lệch phải **lớn hơn độ lệch chuẩn** mới đáng tin.
Đọc kỹ các dòng `!` trong output: nhóm chiếm quá nửa tập đo, tập đo thiếu lớp, lớp dưới 10 mẫu.

**Kỷ luật holdout** (quan trọng nhất của vòng B):
- Hằng ngày chỉ đo trên **dev / CV**. `train` mặc định không chạm vào holdout.
- `--final` chỉ chạy khi **đã chốt model**, thường là cuối GĐ4 và trước khi viết báo cáo.
- Với dữ liệu người chơi, holdout là **holdout tương lai**: các phiên thu *sau khi* chốt model,
  để trong `data/raw/v1/human_holdout/`. `train` báo lỗi nếu holdout trùng trận với dữ liệu train.
- Kết quả holdout **chỉ để ghi nhận**. Không được xem holdout sai ở đâu rồi quay lại sửa
  feature/model; làm vậy là "tune vào tập kiểm chứng" và con số holdout mất giá trị.
- Không đổi `split.salt` trong config. Test `test_all_configs_share_the_frozen_salt` sẽ chặn.

## 2. Mô hình nhánh

```text
main      ●───────────────●───────────────●          chỉ nhận PR từ develop, luôn gắn tag
          v0.1.0          v0.2.0          v1.0.0
develop ──●──●──●──●──────●──●──●──●──────●          nhánh làm việc chính, luôn xanh
             ▲     ▲         ▲     ▲
feat/… ──────┘     │         │     │                 mọi nhánh bắt đầu từ develop
exp/…  ────────────┘         │     │                 và mở PR về develop
fix/…  ──────────────────────┘     │
```

- **Không commit thẳng** vào `main` hoặc `develop`. Không force-push hai nhánh này.
- Tên nhánh: `<type>/<mo-ta-ngan>`, chữ thường, không dấu, nối bằng `-`.
  Ví dụ: `feat/add-landing-features`, `exp/rf-depth-sweep`, `fix/mirror-ball-speed`.
  Không dùng tên chung chung như `update`, `test`, `temp`.
- **Phát hành lên `main`:** cuối mỗi giai đoạn trong PLAN.md, hoặc trước buổi gặp giảng viên:
  1. Mở PR `develop` → `main`, CI xanh thì merge.
  2. Gắn tag: `git tag -a v0.2.0 -m "GĐ2: model v1 bắt chước bot"` rồi `git push --tags`.
  3. Tạo GitHub Release từ tag, đính kèm file `.joblib` và `.h` của model chốt.
- Gợi ý số phiên bản: `v0.<số giai đoạn>.<lần sửa>`. Bản bảo vệ đồ án là `v1.0.0`.

## 3. Conventional Commits

```text
type(scope): mô tả ngắn ở thể mệnh lệnh
```

- `type`: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `perf`, `exp` (thí nghiệm).
- `scope` bắt buộc, dạng `kebab-case`: `features`, `split`, `train`, `export`, `schema`,
  `data`, `configs`, `docs`, `ci`, `repo`…
- Subject không quá 72 ký tự, không có dấu chấm cuối. Có thể viết tiếng Việt hoặc tiếng Anh,
  nhưng giữ một ngôn ngữ trong suốt repo.
- Cần giải thích lý do hay đánh đổi thì viết vào phần body.

Ví dụ:

```text
feat(features): add landing_dx and landing_on_my_side
fix(features): flip ball speed sign for right-side agent
exp(configs): sweep random forest max_depth 8-16
docs(decisions): record choice of behavior cloning
```

Không đẩy commit kiểu `WIP`, `temp`, `fix later` lên `develop`. Mỗi commit phải qua được `check`.

## 4. Ma trận lan truyền thay đổi

Trước khi sửa, xác định loại thay đổi. Mọi artifact trong cột giữa phải được cập nhật
**trong cùng PR**.

| Loại thay đổi | Phải cập nhật cùng lúc | Kiểm tra |
|---|---|---|
| Feature / schema log | `schema/*.json` **trước** → `features.py` → `tests/` → `docs/data_contract.md` → báo bạn làm game | `check` (test thứ tự feature khớp spec) |
| Cách chia / cách đo | `split.py` / `evaluate.py` / `train.py` → `tests/test_split.py`, `test_evaluate.py`, `test_train.py` → mục 1 tài liệu này → ADR mới nếu đổi chiến lược; ghi rõ holdout cũ còn dùng được không; ghi chú các dòng cũ trong `experiments.md` nếu không còn so sánh được | `check` |
| Model đưa vào game | export `.h` → `RunGoldenTest()` pass bên game → `docs/integration.md` nếu cách gọi đổi → ghi run ID vào `WORKLOG.md` | golden test Python + C++ |
| Thêm model mới | `models.py` (và `EXPORTABLE` nếu export được) → test export nếu export được | `check` |
| Dependency | `requirements.txt` (chạy pipeline) hoặc `requirements-dev.txt` (test, lint, notebook) | CI cài lại từ đầu |
| Quyết định kỹ thuật lớn | ADR mới trong `docs/decisions/` | review |
| Quy trình làm việc | chỉ sửa file này; `AGENTS.md` và README chỉ trỏ tới đây | — |

## 5. Definition of Done

Một việc chỉ được coi là **xong** khi:

- [ ] `scripts/check.ps1` PASS **ở lần chạy cuối cùng** (không dựa vào kết quả cũ).
- [ ] Behavior mới có test. Bug fix có test tái hiện bug.
- [ ] Không xoá, bỏ qua hay nới lỏng test chỉ để cho qua.
- [ ] Docs đã cập nhật theo ma trận ở mục 4.
- [ ] CI xanh trên PR.
- [ ] `WORKLOG.md` có dòng mới nếu hoàn thành một chức năng hoặc outcome.
- [ ] Plan (nếu có) đã điền tổng kết và chuyển sang `docs/plans/completed/`.
- [ ] Không commit dữ liệu, model, secret hay file tạm.

## 6. Nơi ghi kết quả

| Ở đâu | Ghi gì | Git |
|---|---|---|
| `models/<run>/` | Mọi thứ của một lần train: model, metrics, split, confusion matrix | không theo dõi |
| `reports/experiments.md` | Một dòng cho mỗi (tập đo, target, model) mỗi lần train, kèm ghi chú | theo dõi |
| `reports/results/<YYYY-MM-DD>/` | Kết quả **đã chốt** để đưa vào báo cáo: bảng, hình, nhận xét. Mỗi ngày giữ một bản cho mỗi chủ đề | theo dõi |
| `WORKLOG.md` | Chức năng / outcome đã xong, có bằng chứng | theo dõi |
| `JOURNAL.md` | Nhật ký tuần: mục tiêu, khó khăn, bài học. Dùng khi gặp giảng viên | theo dõi |
| `docs/decisions/` | Quyết định kỹ thuật và lý do (ADR) | theo dõi |
| `docs/plans/` | Kế hoạch từng việc, đang làm và đã xong | theo dõi |

## 7. Lệnh hay dùng

```bash
# kiểm tra trước khi commit
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1
# tự sửa lỗi lint đơn giản và định dạng code
.venv\Scripts\python -m ruff check . --fix
.venv\Scripts\python -m ruff format .
# chạy một nhóm test
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1 -k split
```
