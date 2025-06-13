import os

from manim import *

os.environ["PATH"] += os.pathsep + r"C:\Users\Misha\AppData\Local\Programs\MiKTeX\miktex\bin\x64"


class UltimateVisualGuide(Scene):
    def construct(self):
        """Полный гид по визуализации в Manim"""

        # ====== 1. НАСТРОЙКА СЦЕНЫ ======
        self.camera.background_color = "#1e1e1e"  # Тёмный фон

        # Анимация заголовка
        title = Text("Визуализация функции cos", font_size=48, color=WHITE)
        self.play(Write(title), run_time=1.5)
        self.wait(0.5)
        self.play(title.animate.to_edge(UP))

        '''# ====== 3. ГРАФИКИ ФУНКЦИЙ ======
        axes = Axes(
            x_range=[-5, 5],
            y_range=[-5, 5],
            x_length=5,
            y_length=5,
            axis_config={"color": WHITE}
        ).to_edge(DOWN)

        graph = axes.plot(lambda x: cos(x), color=YELLOW)
        area = axes.get_area(graph, x_range=(-pi/2, pi/2), color=BLUE, opacity=0.5)

        self.play(
            Create(axes),
            run_time=1.5
        )
        self.play(Create(graph), run_time=5)
        self.play(FadeIn(area))
        self.wait(1)'''

        # ====== 4. РАБОТА С ТЕКСТОМ ======
        # r"\int_a^b x^2 dx = \frac{b^3}{3} - \frac{a^3}{3}"
        # final_text = Text("Готово!", font_size=72, gradient=(BLUE, GREEN))
        # formula.animate.move_to(ORIGIN).scale(1)
        formula_1 = MathTex(r"1111^4+111^3+11^2+1^1")
        formula_2 = MathTex(r"(111*10+1)^4+(11*10+1)^3+(1*10+1)^2+1^1")
        formula_3 = MathTex(r"x = 1*10+1")
        formula_4 = MathTex(r"(111*10+1)^4+(10x+1)^3+(x)^2+1^1")
        formula_5 = MathTex(r"x_1 = 10x+1")
        formula_6 = MathTex(r"(10x_1+1)^4+(x_1)^3+(x)^2+1^1")
        formula_7 = MathTex(r"(10x_1+1)^4+(x_1^2*x_1)+(x)^2+1^1")
        formula_8 = MathTex(r"((10x_1+1)^2)^2+(x_1^2*x_1)+(x)^2+1^1")
        formula_9 = MathTex(r"((10x_1+1)*(10x_1+1))^2+(x_1^2*x_1)+(x)^2+1^1")
        self.play(Write(formula_1))
        self.wait(2)
        self.play(
            Transform(formula_1, formula_2),
            run_time=2
        )
        self.wait(2)
        self.play(
            FadeOut(formula_1),
            Transform(formula_2, formula_3),
            run_time=2
        )
        self.wait(1)
        self.play(
            FadeOut(formula_2),
            Transform(formula_3, formula_2),
            run_time=2
        )
        self.wait(0.5)
        self.play(
            FadeOut(formula_3),
            Transform(formula_2, formula_4),
            run_time=2
        )
        self.wait(2)
        self.play(
            FadeOut(formula_2),
            Transform(formula_4, formula_5),
            run_time=2
        )
        self.wait(2)
        self.play(
            FadeOut(formula_4),
            Transform(formula_5, formula_6),
            run_time=2
        )


# ====== КАК ЗАПУСКАТЬ ======
"""
1. Предпросмотр (низкое качество, быстро):
manim -pql visualizations.py UltimateVisualGuide

2. HD-качество (1080p):
manim -pqh visualizations.py UltimateVisualGuide

3. Сохранение в MP4:
manim -pqh visualizations.py UltimateVisualGuide --format=mp4
"""
