import math
from collections import defaultdict

problems = []
template_problems = defaultdict(list)  # {(dim, template_name): [problems]}
current_template = None

def add(q, dim, ans):
    global current_template
    # always append to global problems list for backward compatibility
    problems.append((q, dim, ans))
    if current_template:
        template_problems[(dim, current_template)].append((q, dim, ans))

# =========================
# 2D TEMPLATES
# =========================

def gen_2d():
    global current_template
    # Square
    current_template = "2d_square"
    for a in range(1, 30):
        add(f"What is the perimeter of a square with side length {a}?", 2, 4*a)
        add(f"What is the area of a square with side length {a}?", 2, a*a)

    # Rectangle
    current_template = "2d_rectangle"
    for a in range(2, 20):
        for b in range(a+1, 20):
            add(f"What is the perimeter of a rectangle with length {a} and width {b}?", 2, 2*(a+b))
            add(f"What is the area of a rectangle with length {a} and width {b}?", 2, a*b)

    # Right triangle (Pythagorean)
    current_template = "2d_right_triangle"
    triples = [(3,4,5),(5,12,13),(6,8,10),(9,12,15),(8,15,17)]
    for a,b,c in triples:
        add(f"What is the area of a right triangle with legs {a} and {b}?", 2, a*b//2)
        add(f"What is the length of the hypotenuse of a right triangle with legs {a} and {b}?", 2, c)

    # Circle (π multiple)
    current_template = "2d_circle"
    for r in range(1, 20):
        add(f"What multiple of pi is the circumference of a circle with radius {r}?", 2, 2*r)
        add(f"What multiple of pi is the area of a circle with radius {r}?", 2, r*r)

# =========================
# 3D TEMPLATES
# =========================

def gen_3d():
    global current_template
    # Cube
    current_template = "3d_cube"
    for a in range(1, 15):
        add(f"What is the volume of a cube with edge length {a}?", 3, a**3)
        add(f"What is the surface area of a cube with edge length {a}?", 3, 6*a*a)

    # Rectangular prism
    current_template = "3d_rectangular_prism"
    for a in range(1, 10):
        for b in range(a, 10):
            for c in range(b, 10):
                add(
                    f"What is the volume of a rectangular prism with length {a}, width {b}, and height {c}?",
                    3, a*b*c
                )
                add(
                    f"What is the surface area of a rectangular prism with length {a}, width {b}, and height {c}?",
                    3, 2*(a*b + b*c + a*c)
                )

                # space diagonal (integer only)
                s = a*a + b*b + c*c
                if int(math.isqrt(s))**2 == s:
                    add(
                        f"What is the length of the space diagonal of a rectangular prism with length {a}, width {b}, and height {c}?",
                        3, int(math.isqrt(s))
                    )

    # Pyramid
    current_template = "3d_pyramid"
    for base in [12, 24, 36, 48]:
        for h in [3, 6, 9]:
            if (base*h) % 3 == 0:
                add(
                    f"What is the volume of a pyramid with base area {base} and height {h}?",
                    3, (base*h)//3
                )

    # Sphere (π multiple)
    current_template = "3d_sphere"
    for r in range(1, 20):
        add(
            f"What multiple of pi is the surface area of a sphere with radius {r}?",
            3, 4*r*r
        )
        if r % 3 == 0:
            add(
                f"What multiple of pi is the volume of a sphere with radius {r}?",
                3, (4*r*r*r)//3
            )

# =========================
# 4D TEMPLATES
# =========================

def gen_4d():
    global current_template
    # Tesseract
    current_template = "4d_tesseract"
    for a in range(1, 10):
        add(
            f"What is the hyper-volume of a tesseract with edge length {a}?",
            4, a**4
        )
        add(
            f"What is the hyper-surface volume of a tesseract with edge length {a}?",
            4, 8*a**3
        )

    # Rectangular 4-parallelotope
    current_template = "4d_rect_par"
    for a in range(1, 6):
        for b in range(a, 6):
            for c in range(b, 6):
                for d in range(c, 6):
                    add(
                        f"What is the hyper-volume of a rectangular 4-parallelotope with length {a}, width {b}, height {c}, and depth {d}?",
                        4, a*b*c*d
                    )
                    add(
                        f"What is the hyper-surface volume of a rectangular 4-parallelotope with length {a}, width {b}, height {c}, and depth {d}?",
                        4, 2*(a*b*c + a*b*d + a*c*d + b*c*d)
                    )

                    # 4D diagonal
                    s = a*a + b*b + c*c + d*d
                    if int(math.isqrt(s))**2 == s:
                        add(
                            f"What is the length of the 4D diagonal of a rectangular 4-parallelotope with length {a}, width {b}, height {c}, and depth {d}?",
                            4, int(math.isqrt(s))
                        )

    # 4-simplex
    current_template = "4d_4simplex"
    for base in [24, 48, 72, 96]:
        for h in [2, 4, 6, 8]:
            if (base*h) % 4 == 0:
                add(
                    f"What is the hyper-volume of a 4-simplex with base volume {base} and height {h}?",
                    4, (base*h)//4
                )

    # 3-sphere (π² multiple)
    current_template = "4d_3sphere"
    for r in range(1, 15):
        add(
            f"What multiple of pi^2 is the hyper-surface volume of a 3-sphere with radius {r}?",
            4, 2*r**3
        )
        if r % 2 == 0:
            add(
                f"What multiple of pi^2 is the hyper-volume of a 3-sphere with radius {r}?",
                4, (r**4)//2
            )

# =========================
# GENERATE & TRIM
# =========================

gen_2d()
gen_3d()
gen_4d()

# Balance sampling: aim for TARGET_PER_DIM per dimension, distributing evenly across templates
TARGET_PER_DIM = 100
final = []
for dim in (2, 3, 4):
    # collect all templates for this dimension
    keys = [k for k in template_problems.keys() if k[0] == dim]
    if not keys:
        continue
    n_templates = len(keys)
    # round-robin take from each template so each contributes similarly
    selected = []
    indices = {k: 0 for k in keys}
    while len(selected) < TARGET_PER_DIM:
        any_added = False
        for key in keys:
            items = template_problems.get(key, [])
            idx = indices.get(key, 0)
            if idx < len(items):
                selected.append(items[idx])
                indices[key] = idx + 1
                any_added = True
                if len(selected) >= TARGET_PER_DIM:
                    break
        if not any_added:
            # no more items available across templates
            break
    final_dim = selected
    final.extend(final_dim)

# =========================
# OUTPUT
# =========================

print("question,dimension,answer")
for q,d,a in final:
    print(f"\"{q}\",{d},{a}")
