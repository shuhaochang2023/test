import numpy as np
from scipy.optimize import linprog
from itertools import product

# Coefficients
coeff_x = [-8160, -62.4, -6750, -6943.1, -118219.5, -249949.8, -214.6, -7727.4, 7892, 124107, 1125436, 8080]
coeff_y = [13040, -62.4, -6750, -6943.1, 223780.5, 434050.2, -214.6, 13472.6, -13308, -217893, -2074564, -13120]
coeff_z = [34240, -62.4, 164250, 164057, 223780.5, 776050.2, 5085.4, 24072.6, -29208, -388893, -2874564, -34320]
coeff_w = [55440, 56937.6, 335250, 506057, 52780.5, 776050.2, 15685.4, 24072.6, -39808, -445893, -2874564, -55520]

names = list("ABCDEFGHIJKL")

def calc(coeffs, values):
    return sum(c * v for c, v in zip(coeffs, values))

def check(vals):
    X = calc(coeff_x, vals)
    Y = calc(coeff_y, vals)
    Z = calc(coeff_z, vals)
    W = calc(coeff_w, vals)
    return X > 9700 and Y > -22400 and Z > 3600 and W > 150000

# Strategy: use large margin, then try rounding each variable up/down
A_ub = [
    [-c for c in coeff_x],
    [-c for c in coeff_y],
    [-c for c in coeff_z],
    [-c for c in coeff_w],
]

# Try increasing margins
for margin in [2000, 5000, 10000, 20000]:
    b_ub = [-(9700+margin), (22400-margin), -(3600+margin), -(150000+margin)]
    bounds = [(0, None)] * 12
    c_obj = [1] * 12

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not result.success:
        continue

    exact = result.x
    # Try rounding each to nearest 0.01, both floor and ceil
    # Only non-zero variables matter
    nonzero_idx = [i for i in range(12) if exact[i] > 0.001]

    # Try all combinations of floor/ceil for non-zero vars
    if len(nonzero_idx) <= 10:  # manageable
        floors = [np.floor(exact[i] * 100) / 100 for i in range(12)]
        ceils = [np.ceil(exact[i] * 100) / 100 for i in range(12)]

        best = None
        for combo in product([0, 1], repeat=len(nonzero_idx)):
            candidate = list(floors)
            for j, idx in enumerate(nonzero_idx):
                if combo[j] == 1:
                    candidate[idx] = ceils[idx]

            if all(v >= 0 for v in candidate) and check(candidate):
                s = sum(candidate)
                if best is None or s < best[0]:
                    best = (s, list(candidate))

        if best:
            print(f"=== Found solution with margin={margin} ===")
            vals = best[1]
            for name, v in zip(names, vals):
                if v > 0:
                    print(f"  {name} = {v:.2f}")

            X = calc(coeff_x, vals)
            Y = calc(coeff_y, vals)
            Z = calc(coeff_z, vals)
            W = calc(coeff_w, vals)
            print(f"\n  X = {X:.4f}  (need > 9700)   {'OK' if X > 9700 else 'FAIL'}")
            print(f"  Y = {Y:.4f}  (need > -22400) {'OK' if Y > -22400 else 'FAIL'}")
            print(f"  Z = {Z:.4f}  (need > 3600)   {'OK' if Z > 3600 else 'FAIL'}")
            print(f"  W = {W:.4f}  (need > 150000) {'OK' if W > 150000 else 'FAIL'}")
            break

    print(f"  margin={margin}: {len(nonzero_idx)} nonzero vars: {[names[i] for i in nonzero_idx]}, exact values: {[round(exact[i],4) for i in nonzero_idx]}")
else:
    print("No rounded solution found with margin approach. Trying different strategy...")

    # Try maximizing slack: maximize min(X-9700, Y+22400, Z-3600, W-150000)
    # Add auxiliary variable s, maximize s
    # X - 9700 >= s => -X + s <= -9700
    # Y + 22400 >= s => -Y + s <= 22400  (wait, Y > -22400 means Y+22400 > 0)
    # Z - 3600 >= s => -Z + s <= -3600
    # W - 150000 >= s => -W + s <= -150000
    # 13 variables: A..L + s (index 12), s unbounded

    c_obj2 = [0]*12 + [-1]  # maximize s = minimize -s
    A_ub2 = [
        [-coeff_x[i] for i in range(12)] + [1],   # -X + s <= -9700
        [-coeff_y[i] for i in range(12)] + [1],   # -Y + s <= 22400
        [-coeff_z[i] for i in range(12)] + [1],   # -Z + s <= -3600
        [-coeff_w[i] for i in range(12)] + [1],   # -W + s <= -150000
    ]
    b_ub2 = [-9700, 22400, -3600, -150000]
    bounds2 = [(0, None)]*12 + [(None, None)]

    result2 = linprog(c_obj2, A_ub=A_ub2, b_ub=b_ub2, bounds=bounds2, method='highs')
    if result2.success:
        exact = result2.x[:12]
        s_val = result2.x[12]
        print(f"Max-min slack solution: s = {s_val:.4f}")

        nonzero_idx = [i for i in range(12) if exact[i] > 0.001]
        print(f"Nonzero vars: {[(names[i], round(exact[i],4)) for i in nonzero_idx]}")

        floors = [np.floor(exact[i] * 100) / 100 for i in range(12)]
        ceils = [np.ceil(exact[i] * 100) / 100 for i in range(12)]

        if len(nonzero_idx) <= 15:
            best = None
            for combo in product([0, 1], repeat=len(nonzero_idx)):
                candidate = list(floors)
                for j, idx in enumerate(nonzero_idx):
                    if combo[j] == 1:
                        candidate[idx] = ceils[idx]

                if all(v >= 0 for v in candidate) and check(candidate):
                    s = sum(candidate)
                    if best is None or s < best[0]:
                        best = (s, list(candidate))

            if best:
                print(f"\n=== Found solution ===")
                vals = best[1]
                for name, v in zip(names, vals):
                    if v > 0:
                        print(f"  {name} = {v:.2f}")

                X = calc(coeff_x, vals)
                Y = calc(coeff_y, vals)
                Z = calc(coeff_z, vals)
                W = calc(coeff_w, vals)
                print(f"\n  X = {X:.4f}  (need > 9700)   {'OK' if X > 9700 else 'FAIL'}")
                print(f"  Y = {Y:.4f}  (need > -22400) {'OK' if Y > -22400 else 'FAIL'}")
                print(f"  Z = {Z:.4f}  (need > 3600)   {'OK' if Z > 3600 else 'FAIL'}")
                print(f"  W = {W:.4f}  (need > 150000) {'OK' if W > 150000 else 'FAIL'}")
            else:
                print("No rounded feasible combo found")
