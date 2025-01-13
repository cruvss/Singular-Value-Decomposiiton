from manim import *
import numpy as np

class SVDVisualization3D(ThreeDScene):
    def create_matrix_mob(self, matrix, name="", font_size=24):
        matrix_str = f"{name}"
        elements = []
        
        if matrix.ndim == 1:
            elements = [f"{elem:.2f}" for elem in matrix]
        else:
            for row in matrix:
                row_str = ""
                for elem in row:
                    row_str += f"{elem:.2f}  "
                elements.append(row_str.strip())
        
        matrix_mob = VGroup(
            Text(matrix_str, font_size=font_size),
            Text("[", font_size=font_size*1.5),
            VGroup(*[Text(elem, font_size=font_size) for elem in elements]).arrange(DOWN, buff=0.2),
            Text("]", font_size=font_size*1.5),
        ).arrange(RIGHT, buff=0.1)
        
        content_height = matrix_mob[2].get_height()
        for bracket in [matrix_mob[1], matrix_mob[3]]:
            bracket.stretch_to_fit_height(content_height)
        
        return matrix_mob

    def construct(self):
        # Create a sample 2x3 matrix for visualization
        A = np.array([[3.0, 1.0, 7.0],
                     [1.0, 2.0, 9.0]])
        U, s, Vt = np.linalg.svd(A)
        
        # Create proper sized Sigma matrix (2x3)
        Sigma = np.zeros(A.shape)
        np.fill_diagonal(Sigma, s)
        
        # Initial setup
        SCALE_FACTOR = 2
        
        # Set up the 3D axes
        axes = ThreeDAxes(
            x_range=[-5, 5],
            y_range=[-5, 5],
            z_range=[-5, 5],
            x_length=10,
            y_length=10,
            z_length=10,
        ).scale(SCALE_FACTOR)
        
        # Add axis labels
        x_label = Text("X").next_to(axes.x_axis, RIGHT)
        y_label = Text("Y").next_to(axes.y_axis, UP)
        z_label = Text("Z").next_to(axes.z_axis, OUT)
        labels = VGroup(x_label, y_label, z_label)
        
        # Set initial camera orientation
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.2)
        
        # Title
        title = Text("3D SVD Visualization", font_size=36)
        title.to_edge(UP)
        title.fix_in_frame()  # Make title always visible
        
        # Show initial setup
        self.add_fixed_in_frame_mobjects(title)
        self.play(Create(axes), Create(labels))
        
        # Create basis vectors in 3D
        vec1 = Arrow3D(
            start=np.array([0, 0, 0]),
            end=np.array([1, 0, 0]),
            color=RED,
        )
        vec2 = Arrow3D(
            start=np.array([0, 0, 0]),
            end=np.array([0, 1, 0]),
            color=GREEN,
        )
        vec3 = Arrow3D(
            start=np.array([0, 0, 0]),
            end=np.array([0, 0, 1]),
            color=BLUE,
        )
        
        self.play(Create(vec1), Create(vec2), Create(vec3))
        
        # Show matrix A
        matrix_A = self.create_matrix_mob(A, "A = ")
        matrix_A.to_corner(UL)
        matrix_A.fix_in_frame()
        self.add_fixed_in_frame_mobjects(matrix_A)
        
        # Transform vectors by A
        transformed_vec1 = Arrow3D(
            start=np.array([0, 0, 0]),
            end=np.array([A[0, 0], A[1, 0], 0]),
            color=RED,
        )
        transformed_vec2 = Arrow3D(
            start=np.array([0, 0, 0]),
            end=np.array([A[0, 1], A[1, 1], 0]),
            color=GREEN,
        )
        transformed_vec3 = Arrow3D(
            start=np.array([0, 0, 0]),
            end=np.array([A[0, 2], A[1, 2], 0]),
            color=BLUE,
        )
        
        # Show SVD components
        equals = Text(" = ").next_to(matrix_A, RIGHT)
        equals.fix_in_frame()
        matrix_U = self.create_matrix_mob(U, "U")
        matrix_U.next_to(equals, RIGHT)
        matrix_U.fix_in_frame()
        
        matrix_Sigma = self.create_matrix_mob(Sigma, "Σ")
        matrix_Sigma.next_to(matrix_U, RIGHT)
        matrix_Sigma.fix_in_frame()
        
        matrix_Vt = self.create_matrix_mob(Vt, "V^T")
        matrix_Vt.next_to(matrix_Sigma, RIGHT)
        matrix_Vt.fix_in_frame()
        
        self.add_fixed_in_frame_mobjects(equals, matrix_U, matrix_Sigma, matrix_Vt)
        
        # Transform basis vectors
        self.play(
            Transform(vec1, transformed_vec1),
            Transform(vec2, transformed_vec2),
            Transform(vec3, transformed_vec3),
        )
        self.wait(2)
        
        # Show V^T rotation
        step1_text = Text("Step 1: V^T Rotation", font_size=24)
        step1_text.to_edge(DOWN)
        step1_text.fix_in_frame()
        self.add_fixed_in_frame_mobjects(step1_text)
        matrix_Vt.set_color(YELLOW)
        
        # Show Sigma scaling
        self.wait(2)
        self.play(FadeOut(step1_text))
        step2_text = Text("Step 2: Σ Scaling", font_size=24)
        step2_text.to_edge(DOWN)
        step2_text.fix_in_frame()
        self.add_fixed_in_frame_mobjects(step2_text)
        matrix_Sigma.set_color(YELLOW)
        
        # Show U rotation
        self.wait(2)
        self.play(FadeOut(step2_text))
        step3_text = Text("Step 3: U Rotation", font_size=24)
        step3_text.to_edge(DOWN)
        step3_text.fix_in_frame()
        self.add_fixed_in_frame_mobjects(step3_text)
        matrix_U.set_color(YELLOW)
        
        self.wait(2)
        
        # Final rotation
        self.play(FadeOut(step3_text))
        verification_text = Text("Final Result: A = UΣV^T", font_size=24)
        verification_text.to_edge(DOWN)
        verification_text.fix_in_frame()
        self.add_fixed_in_frame_mobjects(verification_text)
        
        # Stop camera rotation
        self.stop_ambient_camera_rotation()
        
        self.wait(2)