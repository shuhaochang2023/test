import numpy as np
from scipy.optimize import linprog

coeff_x = [-8160, -62.4, -6750, -6943.1, -118219.5, -249949.8, -214.6, -7727.4, 7892, 124107, 1125436, 8080]
coeff_y = [13040, -62.4, -6750, -6943.1, 223780.5, 434050.2, -214.6, 13472.6, -13308, -217893, -2074564, -13120]
coeff_z = [34240, -62.4, 164250, 164057, 223780.5, 776050.2, 5085.4, 24072.6, -29208, -388893, -2874564, -34320]
coeff_w = [55440, 56937.6, 335250, 506057, 52780.5, 776050.2, 15685.4, 24072.6, -39808, -445893, -2874564, -55520]

names_eq = ['X', 'Y', 'Z', 'W']
coeffs = [coeff_x, coeff_y, coeff_z, coeff_w]
# Targets: X>9950, Y>-22200, W>3600, Z>150000
targets = {'X': 9950, 'Y': -22200, 'Z': 150000, 'W': 3600}

bounds = [(0, None)] * 12

# Test each pair of constraints
from itertools import combinations
print("=== Pairwise feasibility ===")
for pair in combinations(names_eq, 2):
    eq_indices = [names_eq.index(p) for p in pair]
    A_ub = []
    b_ub = []
    for idx in eq_indices:
        name = names_eq[idx]
        A_ub.append([-c for c in coeffs[idx]])
        b_ub.append(-targets[name])
    result = linprog([1]*12, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    status = "OK" if result.success else "INFEASIBLE"
    print(f"  {pair[0]} & {pair[1]}: {status}")

# Test each triple
print("\n=== Triple feasibility ===")
for triple in combinations(names_eq, 3):
    eq_indices = [names_eq.index(p) for p in triple]
    A_ub = []
    b_ub = []
    for idx in eq_indices:
        name = names_eq[idx]
        A_ub.append([-c for c in coeffs[idx]])
        b_ub.append(-targets[name])
    result = linprog([1]*12, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    status = "OK" if result.success else "INFEASIBLE"
    print(f"  {' & '.join(triple)}: {status}")

# For infeasible pairs/triples, find the best achievable
print("\n=== Relaxation analysis ===")
# What's the best Z we can achieve while satisfying X>9950, Y>-22200, W>3600?
# Maximize Z subject to X>9950, Y>-22200, W>3600
A_ub_xyw = [
    [-c for c in coeff_x],
    [-c for c in coeff_y],
    [-c for c in coeff_w],
]
b_ub_xyw = [-9950, 22200, -3600]
c_z = [-c for c in coeff_z]  # maximize Z = minimize -Z
r = linprog(c_z, A_ub=A_ub_xyw, b_ub=b_ub_xyw, bounds=bounds, method='highs')
if r.success:
    max_z = sum(coeff_z[i]*r.x[i] for i in range(12))
    print(f"Max Z achievable with X>9950, Y>-22200, W>3600: {max_z:.2f} (need > 150000)")
else:
    print("Even X,Y,W alone might be infeasible, checking...")
    # Check X,Y,W
    r2 = linprog([1]*12, A_ub=A_ub_xyw, b_ub=b_ub_xyw, bounds=bounds, method='highs')
    print(f"  X,Y,W feasibility: {'OK' if r2.success else 'INFEASIBLE'}")

# What's the best X we can achieve while satisfying Y>-22200, W>3600, Z>150000?
A_ub_ywz = [
    [-c for c in coeff_y],
    [-c for c in coeff_w],
    [-c for c in coeff_z],
]
b_ub_ywz = [22200, -3600, -150000]
c_x = [-c for c in coeff_x]
r = linprog(c_x, A_ub=A_ub_ywz, b_ub=b_ub_ywz, bounds=bounds, method='highs')
if r.success:
    max_x = sum(coeff_x[i]*r.x[i] for i in range(12))
    print(f"Max X achievable with Y>-22200, W>3600, Z>150000: {max_x:.2f} (need > 9950)")

# What's the best Y (least negative) with X>9950, W>3600, Z>150000?
A_ub_xwz = [
    [-c for c in coeff_x],
    [-c for c in coeff_w],
    [-c for c in coeff_z],
]
b_ub_xwz = [-9950, -3600, -150000]
c_y = [-c for c in coeff_y]  # maximize Y
r = linprog(c_y, A_ub=A_ub_xwz, b_ub=b_ub_xwz, bounds=bounds, method='highs')
if r.success:
    max_y = sum(coeff_y[i]*r.x[i] for i in range(12))
    print(f"Max Y achievable with X>9950, W>3600, Z>150000: {max_y:.2f} (need > -22200)")

# What's the best W with X>9950, Y>-22200, Z>150000?
A_ub_xyz = [
    [-c for c in coeff_x],
    [-c for c in coeff_y],
    [-c for c in coeff_z],
]
b_ub_xyz = [-9950, 22200, -150000]
c_w = [-c for c in coeff_w]  # maximize W
r = linprog(c_w, A_ub=A_ub_xyz, b_ub=b_ub_xyz, bounds=bounds, method='highs')
if r.success:
    max_w = sum(coeff_w[i]*r.x[i] for i in range(12))
    print(f"Max W achievable with X>9950, Y>-22200, Z>150000: {max_w:.2f} (need > 3600)")
