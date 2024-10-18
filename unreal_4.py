import sympy as sp
import numpy as np
from fractions import Fraction
import itertools
import copy


points = eval("[(Fraction(400, 1), Fraction(1200, 1)), (Fraction(1200, 1), Fraction(1200, 1)), (Fraction(800, 1), Fraction(1200, 1)), (Fraction(800, 1), Fraction(1700, 1)), (Fraction(800, 1), Fraction(200, 1)), (Fraction(800, 1), Fraction(700, 1)), (Fraction(600, 1), Fraction(1200, 1)), (Fraction(26434, 29), Fraction(12215, 29)), (Fraction(24814, 29), Fraction(16265, 29)), (Fraction(1000, 1), Fraction(1200, 1)), (Fraction(269176957846, 361068821), Fraction(26398762085, 361068821)), (Fraction(800, 1), Fraction(658100, 2093))]")
point_pairs = eval("[(2, 3), (5, 2), (6, 2), (6, 0), (5, 6), (8, 7), (8, 5), (9, 2), (9, 1), (8, 9), (11, 5), (11, 4), (11, 8), (11, 10)]")

points = eval("[(Fraction(400, 1), Fraction(800, 1)), (Fraction(800, 1), Fraction(750, 1)), (Fraction(600, 1), Fraction(400, 1))]")
point_pairs = [(0, 1), (1, 2), (2, 0)]

points = [(100, 400), (300, 400), (300, 200), (Fraction(200, 1), Fraction(300, 1)), (Fraction(200, 1), Fraction(400, 1))]
point_pairs = [(1, 2), (3, 2), (3, 0), (3, 1), (4, 0), (4, 1), (4, 3)]

points = [(100, 400), (300, 400), (300, 200), (Fraction(200, 1), Fraction(300, 1)), (Fraction(200, 1), Fraction(400, 1)), (Fraction(150, 1), Fraction(350, 1)), (Fraction(200, 1), Fraction(1100, 3))]
point_pairs = [(1, 2), (3, 2), (3, 1), (4, 0), (4, 1), (5, 3), (5, 0), (6, 4), (6, 3), (6, 1), (6, 5)]

points = [(100, 400), (300, 400), (300, 200), (Fraction(200, 1), Fraction(300, 1)), (Fraction(200, 1), Fraction(400, 1))]
point_pairs = [(1, 2), (3, 2), (3, 0), (3, 1), (4, 0), (4, 1), (4, 3)]

points = [(100, 400), (300, 400), (300, 200), (Fraction(200, 1), Fraction(300, 1)), (Fraction(200, 1), Fraction(400, 1)), (Fraction(29, 1), Fraction(471, 1))]
point_pairs = [(1, 2), (3, 2), (3, 0), (3, 1), (4, 0), (4, 1), (4, 3), (0, 5)]

points = [(100, 400), (300, 400), (300, 200), (Fraction(200, 1), Fraction(300, 1)), (Fraction(200, 1), Fraction(400, 1)), (Fraction(29, 1), Fraction(471, 1)), (Fraction(371, 1), Fraction(471, 1)), (Fraction(400, 1), Fraction(400, 1))]
point_pairs = [(1, 2), (3, 2), (3, 0), (3, 1), (4, 0), (4, 1), (4, 3), (0, 5), (1, 6), (1, 7)]

points = [(100, 400), (300, 400), (300, 200), (Fraction(200, 1), Fraction(300, 1)), (Fraction(200, 1), Fraction(400, 1)), (Fraction(29, 1), Fraction(471, 1)), (Fraction(371, 1), Fraction(471, 1)), (Fraction(400, 1), Fraction(400, 1)), (Fraction(0, 1), Fraction(400, 1))]
point_pairs = [(1, 2), (3, 2), (3, 0), (3, 1), (4, 0), (4, 1), (4, 3), (0, 5), (1, 6), (1, 7), (0, 8)]

points = [(100, 400), (300, 400), (300, 200), (Fraction(200, 1), Fraction(300, 1)), (Fraction(200, 1), Fraction(400, 1)), (Fraction(29, 1), Fraction(471, 1)), (Fraction(371, 1), Fraction(471, 1)), (Fraction(400, 1), Fraction(400, 1)), (Fraction(0, 1), Fraction(400, 1)), (Fraction(129, 1), Fraction(229, 1))]
point_pairs = [(1, 2), (3, 2), (3, 0), (3, 1), (4, 0), (4, 1), (4, 3), (0, 5), (1, 6), (1, 7), (0, 8), (3, 9)]

points = [(100, 400), (300, 400), (300, 200), (Fraction(200, 1), Fraction(300, 1)), (Fraction(200, 1), Fraction(400, 1)), (Fraction(29, 1), Fraction(471, 1)), (Fraction(371, 1), Fraction(471, 1)), (Fraction(400, 1), Fraction(400, 1)), (Fraction(0, 1), Fraction(400, 1)), (Fraction(129, 1), Fraction(229, 1)), (Fraction(200, 1), Fraction(500, 1))]
point_pairs = [(1, 2), (3, 2), (3, 0), (3, 1), (4, 0), (4, 1), (4, 3), (0, 5), (1, 6), (1, 7), (0, 8), (3, 9), (4, 10)]

points = [(100, 400), (300, 400), (300, 300), (100, 300), (150, 300), (250, 300), (250, 250), (150, 250), (Fraction(200, 1), Fraction(300, 1)), (Fraction(200, 1), Fraction(250, 1))]
point_pairs = [(0, 1), (1, 2), (3, 0), (2, 5), (3, 4), (4, 7), (5, 6), (8, 4), (8, 5), (9, 7), (9, 6), (9, 8)]

points = [(100, 400), (300, 400), (300, 300), (100, 300), (150, 300), (250, 300), (250, 250), (150, 250), (Fraction(200, 1), Fraction(300, 1)), (Fraction(200, 1), Fraction(250, 1)), (Fraction(0, 1), Fraction(400, 1)), (Fraction(400, 1), Fraction(400, 1))]
point_pairs = [(0, 1), (1, 2), (3, 0), (2, 5), (3, 4), (4, 7), (5, 6), (8, 4), (8, 5), (9, 7), (9, 6), (9, 8)]

points = [(100, 400), (300, 400), (300, 300), (100, 300), (150, 300), (250, 300), (250, 250), (150, 250), (Fraction(200, 1), Fraction(300, 1)), (Fraction(200, 1), Fraction(250, 1)), (Fraction(0, 1), Fraction(400, 1)), (Fraction(400, 1), Fraction(400, 1))]
point_pairs = [(0, 1), (1, 2), (3, 0), (2, 5), (3, 4), (4, 7), (5, 6), (8, 4), (8, 5), (9, 7), (9, 6), (9, 8), (10, 0), (1, 11)]

points = [(100, 400), (300, 400), (300, 300), (100, 300), (150, 300), (250, 300), (250, 250), (150, 250), (Fraction(200, 1), Fraction(300, 1)), (Fraction(200, 1), Fraction(250, 1)), (Fraction(0, 1), Fraction(400, 1)), (Fraction(400, 1), Fraction(400, 1)), (Fraction(200, 1), Fraction(400, 1))]
point_pairs = [(1, 2), (3, 0), (2, 5), (3, 4), (4, 7), (5, 6), (8, 4), (8, 5), (9, 7), (9, 6), (9, 8), (10, 0), (1, 11), (12, 0), (12, 1), (12, 8)]

points = [(100, 400), (300, 400), (300, 300), (100, 300), (150, 300), (250, 300), (250, 250), (150, 250), (Fraction(200, 1), Fraction(300, 1)), (Fraction(200, 1), Fraction(250, 1)), (Fraction(0, 1), Fraction(400, 1)), (Fraction(400, 1), Fraction(400, 1)), (Fraction(200, 1), Fraction(400, 1)), (Fraction(200, 1), Fraction(500, 1))]
point_pairs = [(1, 2), (3, 0), (2, 5), (3, 4), (4, 7), (5, 6), (8, 4), (8, 5), (9, 7), (9, 6), (9, 8), (10, 0), (1, 11), (12, 0), (12, 1), (12, 8), (12, 13)]
def n2a(number):
    return chr(number + ord("A"))
def a2n(letter):
    return ord(letter) - ord("A")
class TreeNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []


def tree_form(tabbed_strings):
    lines = tabbed_strings.split("\n")
    root = TreeNode("Root")
    current_level_nodes = {0: root}
    stack = [root]
    for line in lines:
        level = line.count(" ")
        node_name = line.strip()
        node = TreeNode(node_name)
        while len(stack) > level + 1:
            stack.pop()
        parent_node = stack[-1]
        parent_node.children.append(node)
        current_level_nodes[level] = node
        stack.append(node)
    return root.children[0]


def str_form(node):
    def recursive_str(node, depth=0):
        result = "{}{}".format(" " * depth, node.name)
        for child in node.children:
            result += "\n" + recursive_str(child, depth + 1)
        return result

    return recursive_str(node)

def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    try:
        if x1 == x2:
            if x3 == x4:
                return None, "parallel vertical lines"
            m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else None
            x = x1
            y = m2 * x + (y3 - m2 * x3) if m2 is not None else None
            return (x, y), "intersect" if y is not None else "no intersection"

        if x3 == x4:
            m1 = (y2 - y1) / (x2 - x1)
            x = x3
            y = m1 * x + (y1 - m1 * x1)
            return (x, y), "intersect"

        m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else None
        m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else None

        if m1 == m2:
            return None, "parallel lines"

        if m1 is None:
            x = x1
            y = m2 * x + (y3 - m2 * x3)
        elif m2 is None:
            x = x3
            y = m1 * x + (y1 - m1 * x1)
        else:

            a = m1
            b = y1 - m1 * x1
            c = m2
            d = y3 - m2 * x3
            x = (d - b) / (a - c)
            y = a * x + b

        return (x, y), "intersect"
    except:
        return None, "error"

def find_intersection_line_with_segment(x1, y1, x2, y2, x3, y3, x4, y4):
    def is_within(x1, x2, x):
        return min(x1, x2) <= x <= max(x1, x2)

    # Find intersection point and status
    ans = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
    
    # Assume there's no intersection initially
    output = False

    if ans[1] == "intersect":
        ix, iy = ans[0]
        
        # Check if the intersection point is within the segment bounds
        if is_within(x3, x4, ix) and is_within(y3, y4, iy):
            # Check if the intersection is just touching (at an endpoint)
            if (ix, iy) == (x3, y3) or (ix, iy) == (x4, y4):
                # It's a touch, return False
                output = False
            else:
                # Otherwise, it's a real intersection
                output = True

    return output


def surrounding_angle(given_point):
    def is_enclosed_angle(curr, h1, h2, h3):
        return find_intersection_line_with_segment(
            curr[0], curr[1], h2[0], h2[1], h1[0], h1[1], h3[0], h3[1]
        )

    lst = []
    for line in point_pairs:
        if given_point == line[0]:
            lst.append(points[line[1]])
        elif given_point == line[1]:
            lst.append(points[line[0]])
    lst = list(set(lst))
    new = []
    for item in itertools.permutations(lst):
        if all(
            is_enclosed_angle(points[given_point], item[i], item[i + 1], item[i + 2])
            for i in range(0, len(item) - 2, 1)
        ):
          
            for item2 in item:
              new.append(n2a(points.index(item2)))
            break

    return new

def generate_graph():
    graph = dict()
    for i in range(len(points)):
        tmp = surrounding_angle(i)
        graph[n2a(i)] = tmp
    return graph

def line_sort(line):
    if a2n(line[0]) > a2n(line[1]):
        line = line[1] + line[0]
    return line

def polygon_area(points):
    n = len(points)
    area = 0
    for i in range(n - 1):
        area += points[i][0] * points[i + 1][1] - points[i][1] * points[i + 1][0]
    area += points[-1][0] * points[0][1] - points[-1][1] * points[0][0]
    return abs(area) / 2

def straight_line(point_list):
    global points
    return polygon_area([points[x] for x in point_list]) == 0
graph = generate_graph()

def distance(P1, P2):
    global points
    x1, y1 = points[P1]
    x2, y2 = points[P2]
    return (x2 - x1)**2 + (y2 - y1)**2  # Squared distance for easier comparison

def extend_line(P1, P2):
    global graph
    current_point = P2
    prev_point = P1
    
    # List to store all points tried, starting with P2
    points_tried = [current_point]
    prev_distance = distance(P1, P2)  # Distance from P1 to P2

    while True:
        # Get surrounding points of the current point
        surrounding = graph[n2a(current_point)]
        
        # Find the next point in the same direction that forms a straight line
        next_point = None
        
        for point_label in surrounding:
            point_index = a2n(point_label)
            
            # Skip the point we just came from to avoid backtracking
            if point_index == prev_point:
                continue
            
            # Check if the new point is along the straight line and further away from P1
            if straight_line([P1, P2, point_index]) and distance(P1, point_index) > prev_distance:
                next_point = point_index
                break
        
        if next_point is not None:
            # Update the previous point and distance for the next iteration
            prev_point = current_point
            prev_distance = distance(P1, next_point)
            current_point = next_point
            points_tried.append(current_point)  # Add to the tried points list
        else:
            break  # No valid next point found, stop the loop

    return points_tried, current_point  # Return all points tried and the farthest point





all_line = []
for item in point_pairs:
    P1, P2 = item
    forward_points, forward_end = extend_line(P1, P2)
    backward_points, backward_end = extend_line(P2, P1)
    complete_line = backward_points[1:][::-1] + list(item) + forward_points[1:]
    all_line.append([n2a(x) for x in complete_line])

all_line = [list(x) for i, x in enumerate(all_line) if not any(x == all_line[j] or x[::-1] == all_line[j] for j in range(i))]

line_counter = []
for item in itertools.combinations(range(len(points)), 2):
    if item[0] > item[1]:
        item[0], item[1] = item[1], item[0]
    line = n2a(item[0])+n2a(item[1])
    if any(line[0] in x and line[1] in x for x in all_line):
        line_counter.append(line)

all_angle = []

angle_counter = []
for item in itertools.combinations(point_pairs, 2):
    if len(set(list(item[0])+list(item[1]))) == 3:
        b = list(set(item[0]) & set(item[1]))[0]
        a = list(set(item[0]) - set([b]))[0]
        c = list(set(item[1]) - set([b]))[0]
        new_angle = [extend_line(b, a)[1], b, extend_line(b, c)[1]]
        if a > c:
            a, c=  c, a
        if new_angle[0] > new_angle[2]:
            new_angle[0], new_angle[2] = new_angle[2], new_angle[0]
        new_angle = "".join([n2a(x) for x in new_angle])
        angle = n2a(a)+n2a(b)+n2a(c)
        all_angle.append(angle)
        angle_counter.append(new_angle)

def standard_angle(angle):
    #if not (any(angle[0] in x and angle[1] in x for x in all_line) and any(angle[2] in x and angle[1] in x for x in all_line)):
    #    return None
    a, b, c = a2n(angle[0]), a2n(angle[1]), a2n(angle[2])
    new_angle = [extend_line(b, a)[1], b, extend_line(b, c)[1]]
    if a > c:
        a, c=  c, a
    if new_angle[0] > new_angle[2]:
        new_angle[0], new_angle[2] = new_angle[2], new_angle[0]
    return "".join([n2a(x) for x in new_angle])

angle_counter  = list(set(angle_counter))

line_matrix_eq = []
line_matrix = []

eq_list = []

for angle in angle_counter:
    if straight_line([a2n(x) for x in angle]):
        eq_list.append(sp.Eq(sp.symbols(angle), sp.pi))

        row = [0] * len(line_counter)
        row[line_counter.index("".join(sorted(angle[0]+angle[1])))] = 1
        row[line_counter.index("".join(sorted(angle[1]+angle[2])))] = 1
        row[line_counter.index("".join(sorted(angle[2]+angle[0])))] = -1
        line_matrix.append(row)
        line_matrix_eq.append(0)

def combine(a, b):
    global t_angle
    global points
    global point_pairs

    if a[1] != b[1]:
        return None
    if len(set(a + b)) != 4:
        return None
    r = a[0] + a[2] + b[0] + b[2]
    r = r.replace([x for x in r if r.count(x) == 2][0], "")

    out = list(r[0] + b[1] + r[1])
    if out[0] > out[2]:
        out[0], out[2] = out[2], out[0]

    return "".join(out)
        
for angle in itertools.permutations(angle_counter, 3):
    if combine(angle[0], angle[1]) == angle[2]:
        
        hhh = [
            a2n(h)
            for h in list(set(angle[0] + angle[1] + angle[2]))
            if list(angle[0] + angle[1] + angle[2]).count(h) == 3
        ][0]
        hh = [
            (a2n(h), hhh)
            for h in list(set(angle[0] + angle[1] + angle[2]))
            if list(angle[0] + angle[1] + angle[2]).count(h) == 2
        ]
        orig = copy.deepcopy(point_pairs)
        point_pairs = hh
        hh = surrounding_angle(hhh)
        point_pairs = copy.deepcopy(orig)

        if hh[1] not in angle[2] and straight_line([a2n(x) for x in angle[2]]):
            eq_list.append(sp.Eq(sp.symbols(angle[0])+sp.symbols(angle[1]), sp.symbols(angle[2])))

def angle_sort(angle):
    angle = list(angle)
    if a2n(angle[0]) > a2n(angle[2]):
        angle[0], angle[2] = angle[2], angle[0]
    return "".join(angle)

for angle in itertools.combinations(angle_counter, 2):
    if (
        angle[0][1] == angle[1][1]
        and straight_line([a2n(x) for x in angle[0]])
        and straight_line([a2n(x) for x in angle[1]])
    ):
        tmp1 = angle_sort(angle[1][0] + angle[0][1] + angle[0][2])
        tmp2 = angle_sort(angle[0][0] + angle[1][1] + angle[1][2])
        eq_list.append(sp.Eq(sp.symbols(tmp1), sp.symbols(tmp2)))

        tmp1 = angle_sort(angle[1][2] + angle[0][1] + angle[0][2])
        tmp2 = angle_sort(angle[1][0] + angle[1][1] + angle[0][0])
        eq_list.append(sp.Eq(sp.symbols(tmp1), sp.symbols(tmp2)))
        

def is_reflex_vertex(polygon, vertex_index):
    prev_index = (vertex_index - 1) % len(polygon)
    next_index = (vertex_index + 1) % len(polygon)
    modified_polygon = polygon[:vertex_index] + polygon[vertex_index + 1 :]
    original_area = polygon_area(polygon)
    modified_area = polygon_area(modified_polygon)
    if modified_area <= original_area:
        return False
    else:
        return True


def is_reflex_by_circle(polygon):
    output = []
    for i in range(len(polygon)):
        if is_reflex_vertex(polygon, i):
            output.append(i)
    return output

all_tri = []

cycle = []

def cycle_return(graph, path):
    global cycle
    for item in graph[path[-1]]:
        if item == path[0] and len(path) > 2:
            cycle.append([a2n(x) for x in path])
        elif item not in path:
            cycle_return(graph, path+[item])
for key in graph.keys():
    cycle_return(graph, [key])

nn = []
for item in cycle:
    if set(item) not in [set(x) for x in nn]:
        nn.append(item)

cycle = nn

new_cycle = []
for item in cycle:
    remove_item = []
    for i in range(-2, len(item) - 2, 1):
        if straight_line([item[i], item[i + 1], item[i + 2]]):
            remove_item.append(item[i + 1])
    new_item = item
    for i in range(len(new_item) - 1, -1, -1):
        if new_item[i] in remove_item:
            new_item.pop(i)
    new_cycle.append(new_item)


for x in new_cycle:
    convex_angle = is_reflex_by_circle([points[y] for y in x])
    
    out = []
    v = None
    for i in range(-2, len(x) - 2, 1):
        angle = [x[i], x[i + 1], x[i + 2]]
        tmp = [[z for z in x][y] for y in convex_angle]

        v = "".join([n2a(y) for y in angle])
        if angle[1] in tmp:
            out.append(
                "(360-" + standard_angle("".join([n2a(y) for y in angle])) + ")"
            )
        else:
            out.append(standard_angle("".join([n2a(y) for y in angle])))

    if len(x) == 3:
        all_tri.append(v)
    if out == []:
        continue
    copy_out = copy.deepcopy(out)

    out = copy.deepcopy(copy_out)
    for i in range(len(out)):
        out[i] = out[i].replace("(360-", "").replace(")", "")

    subtract = 0
    eq_curr = 0
    for i in range(len(out)):
        if "(360-" in copy_out[i]:

            subtract += sp.pi*2
            eq_curr += -sp.symbols(out[i])
            
        else:
            eq_curr += sp.symbols(out[i])
    eq_list.append(sp.Eq(eq_curr, sp.pi * (len(x) - 2) - subtract))

all_tri = list(set(all_tri))

#######


for index, item in enumerate(line_matrix):
    if (
        item.count(0) == len(item) - 2
        and item.count(1) == 1
        and item.count(-1) == 1
        and line_matrix_eq[index] == 0
    ):
        line1 = line_counter[item.index(1)]
        line2 = line_counter[item.index(-1)]
        for tri in all_tri:
            if (
                line1[0] in tri
                and line2[0] in tri
                and line1[1] in tri
                and line2[1] in tri
            ):
                common = set(line1) & set(line2)
                common = list(common)[0]
                a = list(set(line1) - set(common))[0]
                b = list(set(line2) - set(common))[0]
                eq_list.append(sp.Eq(sp.symbols(standard_angle(common + a + b)), sp.symbols(standard_angle(common + b + a))))

                break

def line_matrix_print():
    def remove_duplicates(matrix_2d, array_1d):
        unique_rows = {}
        for row, val in zip(matrix_2d, array_1d):
            row_tuple = tuple(row)
            if row_tuple not in unique_rows:
                unique_rows[row_tuple] = val
        new_matrix_2d = list(unique_rows.keys())
        new_array_1d = list(unique_rows.values())
        return new_matrix_2d, new_array_1d

    global line_matrix
    global line_matrix_eq
    global line_counter
    
    line_matrix, line_matrix_eq = remove_duplicates(line_matrix, line_matrix_eq)
    for i in range(len(line_matrix)):
        line_matrix[i] = list(line_matrix[i])

    string = "$"
    for i in range(len(line_matrix)):
        for j in range(len(line_matrix[i])):
            if line_matrix[i][j] != 0:
                if line_matrix[i][j] == -1:
                    string += "-line(" + line_counter[j] + ")"
                elif line_matrix[i][j] == 1:
                    string += "+line(" + line_counter[j] + ")"
                else:
                    string += (
                        "+" + str(line_matrix[i][j]) + "*line(" + line_counter[j] + ")"
                    )
        string += "=" + str(sp.simplify(line_matrix_eq[i])) + "\n"
    string = string.replace("\n+", "\n").replace("$+", "").replace("$", "")
    if string != "":
        print(string)

def line_eq(line1, line2):
    if line1 == line2:
        return True

    row = [0] * len(line_counter)
    row[line_counter.index(line1)] = 1
    row[line_counter.index(line2)] = -1
    if row in line_matrix and line_matrix_eq[line_matrix.index(row)] == 0:
        return True
    row[line_counter.index(line1)] = -1
    row[line_counter.index(line2)] = 1
    if row in line_matrix and line_matrix_eq[line_matrix.index(row)] == 0:
        return True
    return False


def angle_eq(angle1, angle2):
    if angle1 == angle2:
        return True

    row = [0] * len(angle_counter)
    row[angle_counter.index(angle1)] = 1
    row[angle_counter.index(angle2)] = -1
    if row in matrix and matrix_eq[matrix.index(row)] == 0:
        return True
    row[angle_counter.index(angle1)] = -1
    row[angle_counter.index(angle2)] = 1
    if row in matrix and matrix_eq[matrix.index(row)] == 0:
        return True
    return False


def angle_per(angle):
    row = [0] * len(angle_counter)
    row[angle_counter.index(angle)] = 1
    if row in matrix and matrix_eq[matrix.index(row)] == sp.pi/2:
        return True
    return False

def sss_rule(a1, a2, a3, b1, b2, b3):
    line = [
        line_sort(a1 + a2),
        line_sort(b1 + b2),
        line_sort(a2 + a3),
        line_sort(b2 + b3),
        line_sort(a1 + a3),
        line_sort(b1 + b3),
    ]

    for item in line:
        if item not in line_counter:
            return False

    return (
        line_eq(line[0], line[1])
        and line_eq(line[2], line[3])
        and line_eq(line[4], line[5])
    )


def sas_rule(a1, a2, a3, b1, b2, b3):
    line = [
        line_sort(a1 + a2),
        line_sort(b1 + b2),
        line_sort(a2 + a3),
        line_sort(b2 + b3),
    ]
    angle = [standard_angle(a1 + a2 + a3), standard_angle(b1 + b2 + b3)]

    for item in line:
        if item not in line_counter:
            return False
    for item in angle:
        if item not in angle_counter:

            return False

    return (
        line_eq(line[0], line[1])
        and angle_eq(angle[0], angle[1])
        and line_eq(line[2], line[3])
    )


def aas_rule(a1, a2, a3, b1, b2, b3):
    line = [line_sort(a2 + a3), line_sort(b2 + b3)]
    angle = [
        standard_angle(a1 + a2 + a3),
        standard_angle(b1 + b2 + b3),
        standard_angle(a3 + a1 + a2),
        standard_angle(b3 + b1 + b2),
    ]

    for item in line:
        if item not in line_counter:
            return False

    for item in angle:
        if item not in angle_counter:
            return False

    return (
        line_eq(line[0], line[1])
        and angle_eq(angle[0], angle[1])
        and angle_eq(angle[2], angle[3])
    )


def rhs_rule(a1, a2, a3, b1, b2, b3):
    line = [
        line_sort(a1 + a2),
        line_sort(b1 + b2),
        line_sort(a1 + a3),
        line_sort(b1 + b3),
    ]
    angle = [standard_angle(a1 + a2 + a3), standard_angle(b1 + b2 + b3)]

    for item in line:
        if item not in line_counter:
            return False

    for item in angle:
        if item not in angle_counter:
            return False

    return (
        line_eq(line[0], line[1])
        and angle_eq(angle[0], angle[1])
        and line_eq(line[2], line[3])
        and angle_per(angle[0])
    )

def line_fx(line_input):
    a = line_input[0]
    b = line_input[1]
    return TreeNode("f_line", [tree_form("d_" + a), tree_form("d_" + b)])

def proof_fx_3(angle1, angle2):
    global eq_list

    angle_1 = TreeNode(
        "f_triangle",
        [
            tree_form("d_" + angle1[0]),
            tree_form("d_" + angle1[1]),
            tree_form("d_" + angle1[2]),
        ],
    )
    angle_2 = TreeNode(
        "f_triangle",
        [
            tree_form("d_" + angle2[0]),
            tree_form("d_" + angle2[1]),
            tree_form("d_" + angle2[2]),
        ],
    )
    eq = TreeNode("f_congruent", [angle_1, angle_2])
    eq = str_form(eq)

    for angle in [angle1 + angle2, angle2 + angle1]:
        if sss_rule(*angle) or sas_rule(*angle) or aas_rule(*angle) or rhs_rule(*angle):
            eq_list.append(eq)
            do_cpct()
            return eq
    return None

def proof_fx_2(a, b):
    global eq_list
    global matrix
    global matrix_eq
    
    u, v = a, b
    for item in itertools.combinations(point_pairs, 2):
        if len(set([item[0][0], item[0][1], item[1][0], item[1][1]])) == 4:
            for item2 in itertools.product(item[0], item[1]):
                if (
                    line_sort(n2a(item2[0]) + n2a(item2[1])) in line_counter
                    and line_sort(n2a(item2[0]) + n2a(item2[1])) != line_sort(u)
                    and line_sort(n2a(item2[0]) + n2a(item2[1])) != line_sort(v)
                ):
                    c = None
                    d = None
                    if item[0][0] in item2:
                        c = item[0][1]
                    if item[0][1] in item2:
                        c = item[0][0]
                    if item[1][0] in item2:
                        d = item[1][1]
                    if item[1][1] in item2:
                        d = item[1][0]
                    a, b = item2
                    if (
                        is_same_line(line_sort(n2a(a) + n2a(c)), line_sort(u))
                        and is_same_line(line_sort(n2a(b) + n2a(d)), line_sort(v))
                    ) or (
                        is_same_line(line_sort(n2a(a) + n2a(c)), line_sort(v))
                        and is_same_line(line_sort(n2a(b) + n2a(d)), line_sort(u))
                    ):
                        tmp = find_intersection_3(
                            points[c][0],
                            points[c][1],
                            points[d][0],
                            points[d][1],
                            points[a][0],
                            points[a][1],
                            points[b][0],
                            points[b][1],
                        )
                        if tmp[1] == "intersect":
                            add_angle_equality(
                                n2a(c) + n2a(a) + n2a(b), n2a(d) + n2a(b) + n2a(a)
                            )

def gauss_jordan_elimination(matrix):
    # Convert the NumPy matrix to a sympy Matrix
    sympy_matrix = sp.Matrix(matrix)
    # Perform Gauss-Jordan elimination
    
    reduced_matrix = sympy_matrix.rref()[0]

    # Convert the sympy Matrix back to a NumPy array
    return np.array(reduced_matrix)
def matrix_to_list(matrix):
    return [list(item) for item in matrix]
def matrices_equal(matrix1, matrix2):

    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        return False

    for row1, row2 in zip(matrix1, matrix2):
        if any(x != y for x, y in zip(row1, row2)):
            return False

    return True
def line_try_matrix():
    global line_matrix
    global line_matrix_eq
    global eq_list
    global line_counter
    if line_matrix == []:
        return

    A = np.array(line_matrix, dtype=object)
    B = np.array(line_matrix_eq, dtype=object)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    augmented_matrix = np.hstack((A, B))

    new_matrix = gauss_jordan_elimination(augmented_matrix)
    for item in matrix_to_list(new_matrix):

        non_zero_count = sum(1 for x in item[:-1] if x != 0)
        if (
            non_zero_count == 1
            and item.count(1) == 1
            and len(item) == len(item[:-1]) + 1
        ):

            line_matrix.append(item[:-1])
            line_matrix_eq.append(item[-1])
            
def solve_linear_system(augmented_matrix):
    # Define the number of variables
    n = len(augmented_matrix[0]) - 1  # Subtract 1 for the augmented part
    # Create variables
    variables = sp.symbols(f'x0:{n}')  # Creates x0, x1, ..., xn-1
    
    # Create equations from the augmented matrix
    equations = []
    for row in augmented_matrix:
        eq = sp.Eq(sum(row[i] * variables[i] for i in range(n)), row[-1])
        equations.append(eq)
    
    # Solve the equations using linsolve
    solution = sp.linsolve(equations, variables)
    
    return solution
def try_matrix():
    global matrix
    global matrix_eq
    
    if not matrix or not matrix_eq:
        return
    aug = [row + [matrix_eq[i]] for i, row in enumerate(matrix)]
    ans = {}
    n = len(matrix)
    should = sp.symbols(f'x0:{n}')
    
    for i in range(2, 4, 1):
        for item in itertools.combinations(range(len(aug)), i):
            new_aug = []
            for j in item:
                new_aug.append(aug[j].copy())
            
            dup = angle_counter.copy()
            for j in range(len(new_aug[0])-2,-1,-1):
                if all(x[j] == 0 for x in new_aug):
                    for k in range(len(new_aug)):
                        new_aug[k].pop(j)
                    dup.pop(j)
            
            if len(new_aug)+1 == len(new_aug[0]):
                solution = solve_linear_system(new_aug)
                for index, sol in enumerate(list(solution)[0]):
                    if not any(x in sol.free_symbols for x in should):
                        ans[dup[index]] = sol
    for key in ans.keys():
        row = [0]*len(angle_counter)
        row[angle_counter.index(key)] = 1
        matrix.append(row)
        matrix_eq.append(ans[key])
    
def try_matrix_2():
    global matrix
    global matrix_eq
    global eq_list
    global angle_counter
    if matrix == []:
        return

    A = np.array(matrix, dtype=object)
    B = np.array(matrix_eq, dtype=object)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    augmented_matrix = np.hstack((A, B))

    new_matrix = gauss_jordan_elimination(augmented_matrix)

    for item in itertools.combinations(angle_counter, 2):
        row = [0] * len(matrix[0])
        row[angle_counter.index(item[0])] = 1
        row[angle_counter.index(item[1])] = -1
        matrix.append(row)
        matrix_eq.append(0)

        if matrices_equal(
            gauss_jordan_elimination(
                np.hstack(
                    (
                        np.array(matrix, dtype=object),
                        np.array(matrix_eq, dtype=object).reshape(-1, 1),
                    )
                )
            ),
            new_matrix,
        ):

            pass
        else:
            matrix.pop(-1)
            matrix_eq.pop(-1)
def perpendicular_set(point, line):
    for item in all_line:
        if line[0] in item and line[1] in item:
            for item2 in all_line:
                common = list(set(item2)&set(item))
                if point in item2 and common != []:
                    angle1 = standard_angle(line[0]+common[0]+point)
                    angle2 = standard_angle(line[1]+common[0]+point)
                    row = [0] * len(angle_counter)
                    row[angle_counter.index(angle1)] = 1
                    matrix.append(row)
                    matrix_eq.append(sp.pi/2)
                    row = [0] * len(angle_counter)
                    row[angle_counter.index(angle2)] = 1
                    matrix.append(row)
                    matrix_eq.append(sp.pi/2)
#perpendicular_set("D", "AB")
def process_command(string):
    eq_type = string.split(" ")[1]
    if "angle_eq" == eq_type:
        a = standard_angle(string.split(" ")[2])
        b = standard_angle(string.split(" ")[3])

        eq_list.append(sp.Eq(sp.symbols(a), sp.symbols(b)))

    elif "angle_val" == eq_type:
        a = standard_angle(string.split(" ")[2])
        val = string.split(" ")[3]
        eq_list.append(sp.Eq(sp.symbols(a), sp.sympify(val)))

    elif "line_val" == eq_type:
        a = line_sort(string.split(" ")[2])
        val = string.split(" ")[3]
        row = [0] * len(line_counter)
        row[line_counter.index(a)] = 1
        line_matrix.append(row)
        line_matrix_eq.append(sp.sympify(val))
def find_angle_val(angle):
    if angle == 0 or angle == sp.pi:
        return angle
    try:
        angle = standard_angle(angle)
    except:
        return None
    if sp.symbols(angle) not in angle_ans.keys():
        return None
    return angle_ans[sp.symbols(angle)]
def is_valid_expression(expr):
    simplified_expr = sp.simplify(expr)
    # Check if the simplified expression involves infinity (oo, zoo) or NaN
    return not (simplified_expr.has(sp.oo) or simplified_expr.has(sp.zoo) or simplified_expr.has(sp.nan))

process_command("equation angle_val JIF pi/2")
process_command("equation angle_val IJG pi/2")
process_command("equation angle_val KAD pi/2")
process_command("equation angle_val LBC pi/2")
process_command("equation angle_val AMI pi/2")
angle_ans = sp.solve(eq_list)
print(angle_ans)
for triangle in all_tri:
    for item in itertools.combinations(list(triangle), 2):
        a = list(set(triangle)-set([item[0]]))
        b = list(set(triangle)-set([item[1]]))
        line1 = line_sort("".join(a))
        line2 = line_sort("".join(b))
        a = standard_angle(a[0]+item[0]+a[1])
        b = standard_angle(b[0]+item[1]+b[1])
        if sp.symbols(a) not in angle_ans or sp.symbols(b) not in angle_ans:
            continue
        tmp1 = find_angle_val(a)
        tmp2 = find_angle_val(b)
        if tmp1 is not None and tmp2 is not None:
            row = [0]*len(line_counter)
            h = 1/sp.sin(tmp1)
            g = -1/sp.sin(tmp2)
            if is_valid_expression(h) and is_valid_expression(g):
                row[line_counter.index(line1)] = h
                row[line_counter.index(line2)] = g
                line_matrix.append(row)
                line_matrix_eq.append(0)
        
line_try_matrix()
#try_matrix_2()
#matrix_print()
line_matrix_print()


def is_same_line(line1, line2):
    global all_line
    for item in all_line:
        if line1[0] in item and line1[1] in item and line2[0] in item and line2[1] in item:
            return True
    return False

def find_line_val(found):
    index2 = line_counter.index(found)
    for j, item3 in enumerate(matrix_to_list(line_matrix)):
        total_1 = 0
        total_0 = 0
        for i, item2 in enumerate(item3):
            item2 = sp.sympify(str(item2))
            if item2.equals(sp.sympify("1")) and i == index2:
                total_1 += 1
            elif item2.equals(sp.sympify("0")):
                total_0 += 1
        if total_1 == 1 and total_1 + total_0 == len(item3):
            return line_matrix_eq[j]

def gravity(object_name):
    for item2 in all_line:
        for item in [item2[0], item2[-1]]:
            a, b = points[a2n(item[0])][0] - points[a2n(object_name)][0], points[a2n(item[0])][1] - points[a2n(object_name)][1]
            name = None
            if a == 0 and item != object_name:
                name = object_name + item
                if b <= 0:
                    name = item + object_name
            if name is not None:
                return name
def slide_dir(object_name, inc, ground):      
    for item in inc:
        if not is_same_line(item, ground) and straight_line([a2n(object_name), a2n(item[0]), a2n(item[1])]):
            for item2 in item:
                if points[a2n(item2)][1] - points[a2n(object_name)][1] >= 0:
                    return object_name + item2
                

def normal_find(object_name, inc, sd):
    all_point_in_inc = []
    for i in range(len(points)):
        if any(straight_line([i, a2n(x[0]), a2n(x[1])]) for x in inc):
            all_point_in_inc.append(n2a(i))
    sd = sd[::-1]
    for p in range(len(points)):
        p = n2a(p)
        if p not in sd and not straight_line([a2n(p), a2n(sd[0]), a2n(sd[1])]) and p in all_point_in_inc:
            tt= find_angle_val(sd + p)
            if tt is not None and tt.equals(sp.pi/2):
                return object_name + p
def polygon(p, poly):
    for i in range(len(poly)):
        item = poly[i] + poly[(i+1)%len(poly)]
        if straight_line([a2n(item[0]), a2n(item[1]), a2n(p)]):
            return True
    return False
def polygon2(p, poly, poly2):
    for i in range(len(poly)):
        item = poly[i] + poly[(i+1)%len(poly)]
        if straight_line([a2n(item[0]), a2n(item[1]), a2n(p)]):
            if p != item[0]:
                return item[0]
            elif p != item[1]:
                return item[1]
    return polygon2(p, poly2, poly)
def normal(obj_1, obj_2):
    for item in line_counter:
        if polygon(item[0], obj_1) and polygon(item[0], obj_2) and\
           polygon(item[1], obj_1) and not polygon(item[1], obj_2):
            if find_angle_val(polygon2(item[0], obj_1, obj_2)+item) == sp.pi/2:
                return item
    return normal(obj_2, obj_1)[::-1]
            
                    

def vc(line):
    a = points[a2n(line[0])]
    b = points[a2n(line[1])]
    return b[1]-a[1], b[0]-a[0]
def dot(a, b):
    return a[0]*b[0]+a[1]*b[1]
def between(line1, line2):
    if is_same_line(line1, line2):
        if dot(vc(line1),vc(line2)) > 0:
            return 0
        return sp.pi
    
    for item in itertools.permutations(range(len(points)), 3):
        if is_same_line(n2a(item[1])+n2a(item[0]), line1) and is_same_line(n2a(item[2])+n2a(item[1]), line2):
            if dot(vc(n2a(item[1])+n2a(item[0])), vc(line1))>=0 and dot(vc(n2a(item[1])+n2a(item[2])), vc(line2))>=0:
                return standard_angle(n2a(item[0])+n2a(item[1])+n2a(item[2]))
def cross(a, b):
    if a[0]*b[1] - a[1]*b[0] < 0:
        return 1
    else:
        return -1

def find_axis(n, gravity):
    for line in line_counter:
        out = None
        tmp = find_angle_val(between(line, n))
        if sp.cos(tmp) == 0:
            tmp2 =find_angle_val(between(line[::-1], n))
            if dot(vc(line),vc(gravity)) > dot(vc(line[::-1]),vc(gravity)):
                out = line
            else:
                out = line[::-1]
            return out

def physics(object_list, gravity, ground):
    eq_list_axis = [[0, 0] for i in range(len(object_list))]
    eq_list_perpendicular = [[0, 0] for i in range(len(object_list))]
    for item in object_list[:-1]:
        i = object_list.index(item)
        n2 = None
        slide = None
        if object_list[i+1] != ground:
            n2 = normal(object_list[i+1], object_list[i+2])
            slide = None
            if find_angle_val(between(n2, gravity)) in [sp.pi, 0]:
                for line in line_counter:
                    if find_angle_val(between(line, n2)) in [sp.pi/2, -sp.pi/2]:
                        slide = line
                        break
            else:
                slide = find_axis(n2, gravity)
        mass_eq_axis = 0
        mass_eq_perpendicular = 0
        eq_list_axis[i][1] += sp.symbols("m_"+str(i+1))*sp.symbols("a_"+str(i+1))
        for item2 in object_list:
            j = object_list.index(item2)
            if item != item2:
                n = None
                axis = None
                try:
                    n = normal(item, item2)
                    if find_angle_val(between(n, gravity)) in [sp.pi, 0]:
                        for line in line_counter:
                            if find_angle_val(between(line, n)) in [sp.pi/2, -sp.pi/2]:
                                axis = line
                                break
                    else:
                        axis = find_axis(n, gravity)
                except:
                    pass
                if axis is not None:
                    a = find_angle_val(between(axis, gravity)) * cross(vc(axis),vc(gravity))
                    mass_eq_axis = sp.symbols("m_"+str(i+1))*sp.symbols("g")*sp.cos(a)
                    mass_eq_perpendicular = sp.symbols("m_"+str(i+1))*sp.symbols("g")*sp.sin(a)
                if n is not None and axis is not None:
                    b = find_angle_val(between(axis, n)) * cross(vc(axis),vc(n))
                    eq_list_axis[i][0] += sp.symbols("n_"+"".join(sorted(str(i+1)+str(j+1))))*sp.cos(b)
                    eq_list_perpendicular[i][0] += sp.symbols("n_"+"".join(sorted(str(i+1)+str(j+1))))*sp.sin(b)
                c = None
                if axis is not None and slide is not None:
                    c = find_angle_val(between(axis, slide[::-1])) * cross(vc(axis),vc(slide[::-1]))
                if c is not None:
                    eq_list_axis[i][0] += sp.symbols("m_"+str(i+1))*sp.symbols("a_"+str(i+2))*sp.cos(c)
                    eq_list_axis[i][1] += sp.symbols("m_"+str(i+1))*sp.symbols("a_"+str(i+2))*sp.sin(c)

        eq_list_axis[i][0] += mass_eq_axis
        eq_list_perpendicular[i][0] += mass_eq_perpendicular
        print(eq_list_axis[i])
        print(eq_list_perpendicular[i])
        
physics(["HGFE", "ABCD", 'AB'], 'JI', 'AB')
