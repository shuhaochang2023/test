import numpy as np
from scipy.optimize import linprog

# Variables: A, B, C, D, E, F, G, H, I, J, K, L (indices 0-11)
# Coefficients for each equation
# X = sum(coeff_x[i] * var[i])
# Y = sum(coeff_y[i] * var[i])
# Z = sum(coeff_z[i] * var[i])
# W = sum(coeff_w[i] * var[i])

coeff_x = [-8160, -62.4, -6750, -6943.1, -118219.5, -249949.8, -214.6, -7727.4, 7892, 124107, 1125436, 8080]
coeff_y = [13040, -62.4, -6750, -6943.1, 223780.5, 434050.2, -214.6, 13472.6, -13308, -217893, -2074564, -13120]
coeff_z = [34240, -62.4, 164250, 164057, 223780.5, 776050.2, 5085.4, 24072.6, -29208, -388893, -2874564, -34320]
coeff_w = [55440, 56937.6, 335250, 506057, 52780.5, 776050.2, 15685.4, 24072.6, -39808, -445893, -2874564, -55520]

# Constraints (linprog uses <= form, so we negate for >= constraints):
# X > 9700  => -X <= -9700  => sum(-coeff_x[i] * var[i]) <= -9700
# Y > -22400 => -Y <= 22400 => sum(-coeff_y[i] * var[i]) <= 22400
# Z > 3600  => -Z <= -3600  => sum(-coeff_z[i] * var[i]) <= -3600
# W > 150000 => -W <= -150000 => sum(-coeff_w[i] * var[i]) <= -150000

# Use slightly stricter bounds to ensure strict inequality
A_ub = [
    [-c for c in coeff_x],
    [-c for c in coeff_y],
    [-c for c in coeff_z],
    [-c for c in coeff_w],
]

b_ub = [-9701, 22399, -3601, -150001]

# Bounds: all variables >= 0
bounds = [(0, None)] * 12

# Objective: minimize sum of all variables (find a feasible point, keep values small)
c = [1] * 12

result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

if result.success:
    names = list("ABCDEFGHIJKL")
    vals = result.x
    # Round to 2 decimal places
    rounded = [round(v, 2) for v in vals]

    print("=== Solution (rounded to 2 decimals) ===")
    for name, rv, v in zip(names, rounded, vals):
        if rv > 0:
            print(f"  {name} = {rv}  (exact: {v})")

    # Verify with rounded values
    def calc(coeffs, values):
        return sum(c * v for c, v in zip(coeffs, values))

    X_val = calc(coeff_x, rounded)
    Y_val = calc(coeff_y, rounded)
    Z_val = calc(coeff_z, rounded)
    W_val = calc(coeff_w, rounded)

    print(f"\n=== Verification with rounded values ===")
    print(f"  X = {X_val:.4f}  (need > 9700)   {'OK' if X_val > 9700 else 'FAIL'}")
    print(f"  Y = {Y_val:.4f}  (need > -22400) {'OK' if Y_val > -22400 else 'FAIL'}")
    print(f"  Z = {Z_val:.4f}  (need > 3600)   {'OK' if Z_val > 3600 else 'FAIL'}")
    print(f"  W = {W_val:.4f}  (need > 150000) {'OK' if W_val > 150000 else 'FAIL'}")

    # If rounding breaks feasibility, try adjusting
    if not (X_val > 9700 and Y_val > -22400 and Z_val > 3600 and W_val > 150000):
        print("\n=== Rounding broke feasibility, trying with more margin ===")
        b_ub2 = [-9800, 22300, -3700, -150100]
        result2 = linprog(c, A_ub=A_ub, b_ub=b_ub2, bounds=bounds, method='highs')
        if result2.success:
            rounded2 = [round(v, 2) for v in result2.x]
            print("=== Solution with margin (rounded to 2 decimals) ===")
            for name, rv in zip(names, rounded2):
                if rv > 0:
                    print(f"  {name} = {rv}")
            X2 = calc(coeff_x, rounded2)
            Y2 = calc(coeff_y, rounded2)
            Z2 = calc(coeff_z, rounded2)
            W2 = calc(coeff_w, rounded2)
            print(f"\n  X = {X2:.4f}  (need > 9700)   {'OK' if X2 > 9700 else 'FAIL'}")
            print(f"  Y = {Y2:.4f}  (need > -22400) {'OK' if Y2 > -22400 else 'FAIL'}")
            print(f"  Z = {Z2:.4f}  (need > 3600)   {'OK' if Z2 > 3600 else 'FAIL'}")
            print(f"  W = {W2:.4f}  (need > 150000) {'OK' if W2 > 150000 else 'FAIL'}")

            if not (X2 > 9700 and Y2 > -22400 and Z2 > 3600 and W2 > 150000):
                print("\n=== Trying with even more margin ===")
                b_ub3 = [-10000, 22000, -4000, -151000]
                result3 = linprog(c, A_ub=A_ub, b_ub=b_ub3, bounds=bounds, method='highs')
                if result3.success:
                    rounded3 = [round(v, 2) for v in result3.x]
                    print("=== Solution with large margin ===")
                    for name, rv in zip(names, rounded3):
                        if rv > 0:
                            print(f"  {name} = {rv}")
                    X3 = calc(coeff_x, rounded3)
                    Y3 = calc(coeff_y, rounded3)
                    Z3 = calc(coeff_z, rounded3)
                    W3 = calc(coeff_w, rounded3)
                    print(f"\n  X = {X3:.4f}  (need > 9700)   {'OK' if X3 > 9700 else 'FAIL'}")
                    print(f"  Y = {Y3:.4f}  (need > -22400) {'OK' if Y3 > -22400 else 'FAIL'}")
                    print(f"  Z = {Z3:.4f}  (need > 3600)   {'OK' if Z3 > 3600 else 'FAIL'}")
                    print(f"  W = {W3:.4f}  (need > 150000) {'OK' if W3 > 150000 else 'FAIL'}")
else:
    print(f"No solution found: {result.message}")
