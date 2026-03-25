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

# New targets: X>9950, Y>-22200, W>3600, Z>150000
def check(vals):
    X = calc(coeff_x, vals)
    Y = calc(coeff_y, vals)
    Z = calc(coeff_z, vals)
    W = calc(coeff_w, vals)
    return X > 9950 and Y > -22200 and W > 3600 and Z > 150000

def show(vals):
    X = calc(coeff_x, vals)
    Y = calc(coeff_y, vals)
    Z = calc(coeff_z, vals)
    W = calc(coeff_w, vals)
    for name, v in zip(names, vals):
        if v > 0:
            print(f"  {name} = {v:.2f}")
    print()
    print(f"  X = {X:.4f}  (need > 9950)    {'OK' if X > 9950 else 'FAIL'}")
    print(f"  Y = {Y:.4f}  (need > -22200)  {'OK' if Y > -22200 else 'FAIL'}")
    print(f"  W = {W:.4f}  (need > 3600)    {'OK' if W > 3600 else 'FAIL'}")
    print(f"  Z = {Z:.4f}  (need > 150000)  {'OK' if Z > 150000 else 'FAIL'}")

# First check: is the continuous problem even feasible?
print("=== Checking continuous feasibility ===")
A_ub = [
    [-c for c in coeff_x],   # -X <= -9950
    [-c for c in coeff_y],   # -Y <= 22200
    [-c for c in coeff_w],   # -W <= -3600
    [-c for c in coeff_z],   # -Z <= -150000
]
b_ub = [-9950, 22200, -3600, -150000]
bounds = [(0, None)] * 12
c_obj = [1]*12  # minimize sum

result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
print(f"Status: {result.message}")
if result.success:
    exact = result.x
    print("Continuous solution exists:")
    for i, (name, v) in enumerate(zip(names, exact)):
        if v > 0.0001:
            print(f"  {name} = {v:.6f}")

    X = calc(coeff_x, exact)
    Y = calc(coeff_y, exact)
    Z = calc(coeff_z, exact)
    W = calc(coeff_w, exact)
    print(f"\n  X = {X:.4f}")
    print(f"  Y = {Y:.4f}")
    print(f"  W = {W:.4f}")
    print(f"  Z = {Z:.4f}")

    # Now try to find a rounded solution using a smarter approach
    # Use MIP-like: try many random perturbations and rounding strategies
    nonzero_idx = [i for i in range(12) if exact[i] > 0.005]
    print(f"\nNonzero vars: {len(nonzero_idx)}")

    # Exhaustive rounding
    floors = [max(0, np.floor(exact[i] * 100) / 100) for i in range(12)]
    ceils = [np.ceil(exact[i] * 100) / 100 for i in range(12)]

    print("Trying all rounding combos...")
    best = None
    for combo in product([0, 1], repeat=len(nonzero_idx)):
        candidate = list(floors)
        for j, idx in enumerate(nonzero_idx):
            if combo[j] == 1:
                candidate[idx] = ceils[idx]
        if check(candidate):
            s = sum(candidate)
            if best is None or s < best[0]:
                best = (s, list(candidate))

    if best:
        print("=== SOLUTION FOUND (direct rounding) ===")
        show(best[1])
    else:
        print("Direct rounding failed.")

        # Try with larger margin
        print("\n=== Trying with increasing margins ===")
        for margin in [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
            b_ub_m = [-(9950+margin), (22200-margin), -(3600+margin), -(150000+margin)]
            r2 = linprog(c_obj, A_ub=A_ub, b_ub=b_ub_m, bounds=bounds, method='highs')
            if not r2.success:
                print(f"  margin={margin}: infeasible")
                continue

            exact2 = r2.x
            nz2 = [i for i in range(12) if exact2[i] > 0.005]
            floors2 = [max(0, np.floor(exact2[i] * 100) / 100) for i in range(12)]
            ceils2 = [np.ceil(exact2[i] * 100) / 100 for i in range(12)]

            if len(nz2) <= 20:
                found = False
                for combo in product([0, 1], repeat=len(nz2)):
                    candidate = list(floors2)
                    for j, idx in enumerate(nz2):
                        if combo[j] == 1:
                            candidate[idx] = ceils2[idx]
                    if check(candidate):
                        print(f"=== SOLUTION FOUND (margin={margin}) ===")
                        show(candidate)
                        found = True
                        break
                if found:
                    break
                else:
                    print(f"  margin={margin}: {len(nz2)} vars {[names[i] for i in nz2]}, no rounding works")
            else:
                print(f"  margin={margin}: too many nonzero vars ({len(nz2)})")

        else:
            # Try a different objective: maximize the slack on rounding
            print("\n=== Trying slack maximization with rounding awareness ===")
            # Instead of minimizing sum, maximize min slack
            c_obj3 = [0]*12 + [-1]
            A_ub3 = [
                [-coeff_x[i] for i in range(12)] + [1],
                [-coeff_y[i] for i in range(12)] + [1],
                [-coeff_w[i] for i in range(12)] + [1],
                [-coeff_z[i] for i in range(12)] + [1],
            ]
            b_ub3 = [-9950, 22200, -3600, -150000]
            bounds3 = [(0, None)]*12 + [(None, None)]

            r3 = linprog(c_obj3, A_ub=A_ub3, b_ub=b_ub3, bounds=bounds3, method='highs')
            if r3.success:
                print(f"Max slack = {r3.x[12]:.4f}")
                exact3 = r3.x[:12]
                for i, (name, v) in enumerate(zip(names, exact3)):
                    if v > 0.0001:
                        print(f"  {name} = {v:.6f}")

                # Check exact values
                X3 = calc(coeff_x, exact3)
                Y3 = calc(coeff_y, exact3)
                Z3 = calc(coeff_z, exact3)
                W3 = calc(coeff_w, exact3)
                print(f"  X={X3:.2f}, Y={Y3:.2f}, Z={Z3:.2f}, W={W3:.2f}")

else:
    print("The continuous problem is INFEASIBLE - no solution exists even without rounding constraint.")
