# ADR-002: Tự viết exporter C++ cho model cây thay vì dùng m2cgen

- Trạng thái: Accepted
- Ngày: 2026-09-21

## Bối cảnh

Model train bằng Python (scikit-learn) phải chạy **bên trong game C++** (Axmol, build bằng MSVC trên Windows)
mỗi frame. Yêu cầu:
- Không thêm thư viện runtime nặng vào game.
- Tính tất định (deterministic), để giữ được replay và khả năng tái lập của simulation.
- Kiểm chứng được rằng C++ cho **cùng kết quả** với Python.

## Các lựa chọn đã xét

| Lựa chọn | Nhận xét |
|---|---|
| **m2cgen** (sinh code C từ model) | Code sinh ra dùng *compound literal* `(double[]){...}` của C99. MSVC biên dịch C++ không chấp nhận cú pháp này, nên phải tách file `.c` riêng và chỉnh CMake. Thư viện cũng ít được bảo trì từ 2022 |
| ONNX Runtime | Hợp với mạng neural, nhưng thêm dependency lớn vào game. Quá nặng cho một cây vài trăm node |
| Chạy Python bên cạnh, gọi qua socket | Chậm, phá tính tất định, phức tạp khi triển khai |
| **Tự viết exporter** sinh header C++ thuần | Khoảng 200 dòng Python. Cây lưu dạng bảng node, duyệt bằng vòng lặp y hệt sklearn |

## Quyết định

Tự viết `src/spike_ai/export_cpp.py`. Exporter sinh một file `.h` C++17 thuần, không dependency, gồm:
- Bảng node `{left, right, feature, threshold}` và xác suất ở lá, cho từng cây.
- Các hàm `PredictProba`, `PredictIndex`.
- `kFeatureNames`, `kClassNames` để code game gọi đúng thứ tự feature.
- `RunGoldenTest()` chứa 20 vector mẫu lấy từ tập dev, kèm xác suất do sklearn tính sẵn.

Để khớp sklearn tuyệt đối, threshold lưu ở kiểu `double` và so sánh `(double)x_float <= threshold`.
Đây đúng là cách sklearn làm: ép X về float32 rồi so với threshold float64.

Ngay lúc export, exporter tự kiểm tra: một bản Python của thuật toán duyệt bảng phải cho cùng xác suất
với `predict_proba`. Test `test_export_tables_match_sklearn` kiểm tra điều này cho cả Decision Tree và Random Forest.

## Hệ quả

- Chỉ hỗ trợ model dạng cây (`EXPORTABLE = {decision_tree, random_forest}`). Nếu GĐ4 chọn MLP thì cần ADR mới (ví dụ ONNX).
- Random Forest nhiều cây sinh header lớn (50 cây sâu 10 ≈ 1.4 MB), làm chậm thời gian biên dịch.
  Nên ưu tiên cây nông hoặc ít cây nếu độ chính xác tương đương.
- Header chưa được biên dịch thử trên máy làm AI (không có compiler). Lần tích hợp đầu tiên phải gọi `RunGoldenTest()` trong game.
