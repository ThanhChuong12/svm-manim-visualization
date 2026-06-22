# Kế Hoạch Thiết Kế Nội Dung (Storyboard & Script Plan)

Tài liệu này ghi lại chi tiết cấu trúc kịch bản và kế hoạch triển khai của tất cả các scene đã được thiết kế trong dự án **Multibiometric Fusion and SVM Visualization**. Mỗi part được chia thành các phase (giai đoạn) logic để truyền tải câu chuyện từ cơ bản đến nâng cao theo phong cách sư phạm điện ảnh (lấy cảm hứng từ 3Blue1Brown).

---

## 🎬 Part 0: Intro (`part0_intro.py`)
**Mục tiêu:** Mở đầu (Hook) thu hút khán giả, giới thiệu vấn đề cốt lõi của bảo mật sinh trắc học và đưa ra cái nhìn tổng quan về lộ trình (Roadmap).

- **Phase 0: Biometric Scan**
  Mở đầu với lưới giao diện sinh trắc học và hiệu ứng quét tia laser. Các icon sinh trắc học cách điệu (khuôn mặt, vân tay, mống mắt, giọng nói) xuất hiện để giới thiệu bối cảnh.
- **Phase 1: XOR Trap**
  Nhanh chóng đưa người xem từ thực tế vào không gian toán học với bài toán XOR kinh điển (4 cụm điểm chéo nhau). Minh họa sự bất lực của một đường thẳng tuyến tính trong việc phân loại.
- **Phase 2: Kernel Lift (Teaser)**
  Một màn "teaser" mãn nhãn: Nâng các điểm dữ liệu XOR từ không gian 2D lên không gian 3D. Khán giả nhận ra rằng bài toán không thể giải ở 2D lại trở nên dễ dàng ở không gian chiều cao hơn.
- **Phase 3: Title Drop**
  Làm tối nền và thả logo / tiêu đề của bộ phim tài liệu xuống màn hình.
- **Phase 4: Roadmap & Cut**
  Trình bày các phần nội dung chính sẽ đi qua (Roadmap). Chuyển cảnh để bắt đầu bài học.

---

## 🎬 Part 1: Unimodal Biometrics (`part1_unibiometric.py`)
**Mục tiêu:** Phân tích những giới hạn của hệ thống sinh trắc học đơn thức (Unimodal), khái niệm FAR/FRR, và lý do vì sao cần nhiều nguồn dữ liệu (Multimodal).

- **Phase 1: Ideal Setup**
  Minh họa biểu đồ phân phối điểm số hoàn hảo ở dạng 1D. Các điểm "Genuine" (Thật) và "Impostor" (Giả mạo) cách xa nhau hoàn toàn.
- **Phase 2: The Reality (Overlap)**
  Chuyển sang hiện thực: Đồ thị bị xô lệch, phân phối của Genuine và Impostor bị giao nhau (overlap). Không có cách nào tách rời chúng một cách hoàn hảo.
- **Phase 3: Threshold & Trade-offs**
  Vẽ đường ranh giới (Threshold). Kéo thả ranh giới này qua lại để người xem tự cảm nhận được sự đánh đổi (trade-offs) giữa FAR (False Acceptance Rate) và FRR (False Rejection Rate). Bất kể đặt ở đâu cũng sinh ra lỗi.
- **Phase 4: Transition to 2D**
  Xoay trục tọa độ 1D hiện tại và thêm vào một trục thứ hai, mở ra khái niệm kết hợp 2 hệ thống sinh trắc học, dẫn thẳng sang Part 2.

---

## 🎬 Part 2: Score Fusion & Linear SVM (`part2_score_fusion.py`)
**Mục tiêu:** Giới thiệu quá trình dung hợp (Fusion) và dạy cho người xem cách hoạt động của mô hình Support Vector Machine (SVM) tuyến tính.

- **Phase 1: Biometric Vectorization**
  Điểm số từ 2 hệ thống (VD: Vân tay và Khuôn mặt) được kết hợp để tạo thành một vector tọa độ 2D. Các điểm dữ liệu "bắn" vào hệ trục tọa độ mới với hiệu ứng vệt sáng (Vector Flash).
- **Phase 2: The "Math Class" Detour**
  Tạm gác dữ liệu sinh trắc học qua một bên, hạ một lưới tọa độ toán học tĩnh (Toy Dataset) xuống để dạy về thuật toán SVM.
  - Sử dụng các điểm được hardcode để tạo một lề (margin) 45 độ cực rộng.
  - Đường phân chia quét dần vào góc tối ưu và nở rộng lề (`margin_tracker`).
  - *Cinematic Focus:* Khi đụng kịch các Support Vectors, các điểm sáng lên, toàn bộ các điểm thừa xung quanh bị làm mờ đi để nhấn mạnh rằng "SVM chỉ quan tâm tới vùng biên".
- **Phase 3: Application & Rescue**
  Đưa trục tọa độ sinh trắc học trở lại. Áp dụng đường phân chia tuyến tính vừa học để "cứu" hệ thống, giải quyết hoàn hảo bài toán overlap ở 1D nhờ có không gian 2D.
- **Phase 4: The XOR Trap (Evolution of Impostors)**
  Những kẻ tấn công tiến hóa! Kẻ thù ngụy tạo một nửa (chỉ mặt hoặc chỉ vân tay), tạo ra các cụm dữ liệu phân bổ chéo góc (XOR problem). SVM tuyến tính vừa học lập tức thất bại.

---

## 🎬 Part 3: The Kernel Trick (`part3_kernel_lift.py`)
**Mục tiêu:** Giải quyết bài toán XOR ở Part 2 bằng "Phép màu" Kernel Trick. Giới thiệu Radial Basis Function (RBF) Kernel.

- **Phase 1: Nonlinear Siege**
  Minh họa cụm dữ liệu phi tuyến tính (dữ liệu thật bị bao vây thành vòng tròn bởi dữ liệu giả). Đường thẳng đành bất lực trong việc giải vây.
- **Phase 2: The Kernel Lift**
  Thay vì cố gắng uốn cong đường phân chia ở 2D, chúng ta bẻ cong chính *không gian* chứa dữ liệu. Thêm trục $Z$, tạo hàm Gaussian biến đổi các điểm dư liệu thật vọt lên cao như một ngọn đồi.
- **Phase 3: The Laser Shield (3D Hyperplane)**
  Khung cảnh 3D quay tròn mãn nhãn. Một mặt phẳng 2D (Hyperplane màu vàng rực) xuất hiện cắt ngang qua không gian 3D, phân tách hoàn hảo "đỉnh đồi" (dữ liệu thật) ra khỏi "thung lũng" (kẻ giả mạo).
- **Phase 4: Projection Back to 2D**
  Chiếu mặt phẳng cắt 3D này dội ngược lại mặt phẳng 2D ban đầu. Lớp giao cắt tạo thành một vòng tròn bao bọc hoàn hảo cụm dữ liệu thật ở giữa, đánh dấu sự chiến thắng tuyệt đối của phi tuyến tính.

---
*(Tài liệu này đóng vai trò như kịch bản gốc để đối chiếu khi nâng cấp mã nguồn Manim)*
