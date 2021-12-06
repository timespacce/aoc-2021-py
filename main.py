import time
from collections import Counter

import numpy as np


def run():
    # task_1()
    # task_2()
    # task_3()
    # task_4()
    # task_5()
    # task_6()
    # task_7()
    # task_8()
    # task_9()
    # task_10()
    task_11()
    task_12()
    return


def task_1():
    s = open("data/data_task_1", "r")
    rows = s.readlines()
    s.close()
    rows = [float(row.rstrip()) for row in rows]
    length = len(rows)
    increased_count = 0
    for i in range(1, length):
        if rows[i] > rows[i - 1]:
            increased_count += 1
    print("INCREASED COUNT : {0}".format(increased_count))
    return


def task_2():
    s = open("data/data_task_1", "r")
    rows = s.readlines()
    s.close()
    rows = [float(row.rstrip()) for row in rows]
    rows_length, size = len(rows), 3
    windows = np.array([rows[i:i + size] for i in range(0, rows_length, 1) if i <= rows_length - size])
    windows_length = len(windows)
    increased_count = 0
    for i in range(1, windows_length):
        if windows[i].sum() > windows[i - 1].sum():
            increased_count += 1
    print("INCREASED COUNT : {0}".format(increased_count))
    return


def task_3():
    s = open("data/data_task_3", "r")
    rows = s.readlines()
    s.close()

    def transform(row):
        direction, value = row.rstrip().split(" ")
        return (direction[0], float(value))

    rows = [transform(row) for row in rows]
    horizontal, depth = 0, 0
    for v, u in rows:
        if v == 'f':
            horizontal += u
        if v == 'd':
            depth += u
        if v == 'u':
            depth -= u
    print("HORIZONTAL = {0} DEPTH = {1} AREA = {2}".format(horizontal, depth, horizontal * depth))
    return


def task_4():
    s = open("data/data_task_3", "r")
    rows = s.readlines()
    s.close()

    def transform(row):
        direction, value = row.rstrip().split(" ")
        return (direction[0], float(value))

    rows = [transform(row) for row in rows]
    horizontal, depth, aim = 0, 0, 0
    for v, u in rows:
        if v == 'f':
            horizontal += u
            depth += u * aim
        if v == 'd':
            aim += u
        if v == 'u':
            aim -= u
    print("HORIZONTAL = {0} DEPTH = {1} AREA = {2}".format(horizontal, depth, horizontal * depth))
    return


def task_5():
    s = open("data/data_task_5", "r")
    rows = s.readlines()
    s.close()

    def transform(row):
        bits = row.rstrip()
        return list(bits)

    rows = np.array([transform(row) for row in rows])
    bit_width = len(rows[0])
    most_common_bits, least_common_bits = [], []
    for i in range(bit_width):
        lane = rows[:, i]
        most_common = Counter(lane)
        most_common_bit = most_common.most_common(2)[0][0]
        least_common_bit = most_common.most_common(2)[1][0]
        most_common_bits.append(most_common_bit)
        least_common_bits.append(least_common_bit)
    most_common_bits, least_common_bits = ''.join(most_common_bits), ''.join(least_common_bits)
    gamma, epsilon = int(most_common_bits, 2), int(least_common_bits, 2)
    print("GAMMA = {0} EPSILON = {1} : POWER = {2}".format(gamma, epsilon, gamma * epsilon))
    return


def task_6():
    s = open("data/data_task_5", "r")
    rows = s.readlines()
    s.close()

    def transform(row):
        bits = row.rstrip()
        return list(bits)

    rows = np.array([transform(row) for row in rows])
    bit_width = len(rows[0])

    rows_iter = rows.copy()
    for i in range(bit_width):
        lane = rows_iter[:, i]
        most_common = Counter(lane)
        most_common_bit = most_common.most_common(2)[0]
        least_common_bit = most_common.most_common(2)[-1]
        if most_common_bit[0] != least_common_bit[0] and most_common_bit[1] == least_common_bit[1]:
            most_common_bit = '1'
        else:
            most_common_bit = most_common_bit[0]
        most_common_values = rows_iter[lane == most_common_bit]
        rows_iter = most_common_values
    if len(rows_iter) > 1:
        print("ERROR")
    most_common_bits = rows_iter[0]

    rows_iter = rows.copy()
    for i in range(bit_width):
        lane = rows_iter[:, i]
        least_common = Counter(lane)
        most_common_bit = least_common.most_common(2)[0]
        least_common_bit = least_common.most_common(2)[-1]
        if most_common_bit[0] != least_common_bit[0] and most_common_bit[1] == least_common_bit[1]:
            least_common_bit = '0'
        else:
            least_common_bit = least_common_bit[0]
        least_common_values = rows_iter[lane == least_common_bit]
        rows_iter = least_common_values
    if len(rows_iter) > 1:
        print("ERROR")
    least_common_bits = rows_iter[0]

    most_common_bits, least_common_bits = ''.join(most_common_bits), ''.join(least_common_bits)
    ogr, co2sr = int(most_common_bits, 2), int(least_common_bits, 2)
    print("oxygen generator rating = {0} CO2 scrubber rating = {1} : life support rating = {2}".format(ogr, co2sr, ogr * co2sr))
    return


def task_7():
    s = open("data/data_task_7", "r")
    rows = s.readlines()
    s.close()

    def transform_row(row):
        output = list(map(int, row.rstrip().split()))
        return output

    def transform_grid(grid):
        output = list(map(transform_row, grid))
        output = np.array(output)
        return output

    numbers = list(map(int, rows[0].split(",")))
    grids = np.array(rows[1:]).reshape((-1, 6))[:, 1:]
    grids = list(map(transform_grid, grids))
    grids_count = len(grids)
    grids_won = set()

    for x in numbers:
        for grid_idx, grid in enumerate(grids):
            matches = np.where(grid == x)
            grid[matches] = -1
            rows_sum, columns_sum = grid.sum(axis=0), grid.sum(axis=1)
            if len(np.where(rows_sum == -5)[0]) != 0 or len(np.where(columns_sum == -5)[0]) != 0:
                non_marked_numbers = grid[np.where(grid != -1)]
                non_marked_sum = non_marked_numbers.sum()
                print("BOARD {0} : non_marked_sum={1} with number={2} has score={3}".format(grid_idx + 1, non_marked_sum, x, non_marked_sum * x))
                return

    return


def task_8():
    s = open("data/data_task_7", "r")
    rows = s.readlines()
    s.close()

    def transform_row(row):
        output = list(map(int, row.rstrip().split()))
        return output

    def transform_grid(grid):
        output = list(map(transform_row, grid))
        output = np.array(output)
        return output

    numbers = list(map(int, rows[0].split(",")))
    grids = np.array(rows[1:]).reshape((-1, 6))[:, 1:]
    grids = list(map(transform_grid, grids))
    grids_count = len(grids)
    grids_won = set()

    for x in numbers:
        for grid_idx, grid in enumerate(grids):
            matches = np.where(grid == x)
            grid[matches] = -1
            rows_sum, columns_sum = grid.sum(axis=0), grid.sum(axis=1)
            if len(np.where(rows_sum == -5)[0]) != 0 or len(np.where(columns_sum == -5)[0]) != 0:
                non_marked_numbers = grid[np.where(grid != -1)]
                non_marked_sum = non_marked_numbers.sum()
                grids_won.add(grid_idx)
                if len(grids_won) == grids_count:
                    print("BOARD {0} : non_marked_sum={1} with number={2} has score={3}".format(grid_idx + 1, non_marked_sum, x, non_marked_sum * x))
                    return

    return


def task_9():
    s = open("data/data_task_9", "r")
    rows = s.readlines()
    s.close()

    ##
    def transform(row):
        points = row.rstrip().split(" -> ")
        points = [point.split(",") for point in points]
        points = [list(map(int, point)) for point in points]
        points = np.array(points)
        return points

    lines = [transform(row) for row in rows]
    lines = np.array([line for line in lines if line[0][0] == line[1][0] or line[0][1] == line[1][1]])

    dimension = np.max(lines) + 1
    chart = np.zeros((dimension, dimension))

    for line in lines:
        (x1, y1), (x2, y2) = line[0], line[1]
        (x1, y1), (x2, y2) = (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))
        chart[y1:y2 + 1, x1:x2 + 1] += 1

    overlapping_count = len(np.where(chart >= 2)[0])
    print("Overlapping is {0}".format(overlapping_count))
    return


def task_10():
    s = open("data/data_task_9", "r")
    rows = s.readlines()
    s.close()

    ##
    def transform(row):
        points = row.rstrip().split(" -> ")
        points = [point.split(",") for point in points]
        points = [list(map(int, point)) for point in points]
        points = np.array(points)
        return points

    lines = [transform(row) for row in rows]
    accepted_lines = []

    for line in lines:
        c1, c2 = line[0], line[1]
        diff_c1c2 = np.abs(c1 - c2)
        c1xc2 = c1.dot(c2)
        c1_norm, c2_norm = np.linalg.norm(c1), np.linalg.norm(c2)
        c1c2_cos = c1xc2 / (c1_norm * c2_norm)
        cos_rad = np.arccos(c1c2_cos)
        cos_deg = np.rad2deg(cos_rad)
        ##
        horizontal_or_vertical = c1[0] == c2[0] or c1[1] == c2[1]
        diagonal = diff_c1c2[0] == diff_c1c2[1]
        if horizontal_or_vertical:
            accepted_lines.append((line, '1'))
        if diagonal:
            accepted_lines.append((line, '2'))

    dimension = np.max(lines) + 1
    chart = np.zeros((dimension, dimension))

    for line, v in accepted_lines:
        c1, c2 = line[0], line[1]
        du = (c2 - c1)
        u = (du / (np.abs(du) + 1e-37)).astype(np.int32)
        i = 0
        while True:
            xy = c1 + i * u
            chart[xy[1], xy[0]] += 1
            i += 1
            if (xy == c2).all():
                break

    chart = np.int32(chart)
    overlapping_count = len(np.where(chart >= 2)[0])
    print("Overlapping is {0}".format(overlapping_count))

    # for row in chart:
    #     for c in row:
    #         print("{0}".format(c if c > 0 else "."), end ="")
    #     print("")
    return


def task_11():
    s = open("data/data_task_11", "r")
    rows = s.readline()
    s.close()
    ##
    fishes = list(map(int, rows.split(",")))
    simulation_length = 80

    grid = np.zeros(9)
    for fish in fishes:
        grid[fish] += 1

    for j in range(simulation_length):
        x = grid[0]

        for i in np.arange(1, 9):
            grid[i - 1] = grid[i]
        grid[8] = 0

        grid[6] += x
        grid[8] += x

    print("LANTERNFISH = {0} @ {1}".format(simulation_length, grid.sum()))
    return


def task_12():
    s = open("data/data_task_11", "r")
    rows = s.readline()
    s.close()
    ##
    fishes = list(map(int, rows.split(",")))
    simulation_length = 256

    grid = np.zeros(9)
    for fish in fishes:
        grid[fish] += 1

    for j in range(simulation_length):
        x = grid[0]

        for i in np.arange(1, 9):
            grid[i - 1] = grid[i]
        grid[8] = 0

        grid[6] += x
        grid[8] += x

    print("LANTERNFISH = {0} @ {1}".format(simulation_length, grid.sum()))
    return


if __name__ == "__main__":
    begin = time.time()
    run()
    runtime = time.time() - begin
    print("ADVENT in {0} s".format(runtime))
