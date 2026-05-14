from manim import *
from src.constants import *

class IntroScene(Scene):
    def construct(self):
        # 1. Hiển thị Tiêu đề chính
        title = Text("Support Vector Machines", color=WHITE).scale(1.2)
        subtitle = Text("& The Kernel Trick", color=HYPERPLANE_COLOR).scale(0.9)
        subtitle.next_to(title, DOWN)
        
        title_group = VGroup(title, subtitle).center()
        
        # 2. Thông tin tác giả & Trường học
        author_info = VGroup(
            Text("Thực hiện: Lê Hà Thanh Chương", font_size=24),
            Text("Trường ĐH Khoa học Tự nhiên - ĐHQG TP.HCM", font_size=20),
            Text("Môn học: Nhận dạng mẫu (Pattern Recognition)", font_size=20, color=GRAY)
        ).arrange(DOWN, aligned_edge=LEFT).to_corner(DL, buff=0.5)

        # 3. Animation Tiêu đề
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(1)
        
        # 4. Xuất hiện thông tin tác giả
        self.play(FadeIn(author_info, shift=RIGHT))
        self.wait(2)
        
        # 5. Lộ trình video (Roadmap)
        self.play(title_group.animate.scale(0.6).to_edge(UP))
        
        roadmap = BulletList(
            "Linear Separability & Max Margin",
            "The XOR Problem (Non-linear cases)",
            "The Power of Kernel Trick (2D to 3D Mapping)",
            "Comparison: Linear vs. RBF vs. Polynomial",
            buff=0.3
        ).scale(0.7).next_to(title_group, DOWN, buff=1)

        self.play(Write(roadmap), run_time=3)
        self.wait(3)
        
        # 6. Kết thúc cảnh intro - Fade Out toàn bộ
        self.play(FadeOut(title_group), FadeOut(author_info), FadeOut(roadmap))
        self.wait(1)