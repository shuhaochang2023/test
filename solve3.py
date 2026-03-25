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

def show(vals):
    X = calc(coeff_x, vals)
    Y = calc(coeff_y, vals)
    Z = calc(coeff_z, vals)
    W = calc(coeff_w, vals)
    for name, v in zip(names, vals):
        if v > 0:
            print(f"  {name} = {v:.2f}")
    print(f"\n  X = {X:.4f}  {'OK' if X > 9700 else 'FAIL'}")
    print(f"  Y = {Y:.4f}  {'OK' if Y > -22400 else 'FAIL'}")
    print(f"  Z = {Z:.4f}  {'OK' if Z > 3600 else 'FAIL'}")
    print(f"  W = {W:.4f}  {'OK' if W > 150000 else 'FAIL'}")

# Strategy: maximize min slack, but also add small perturbation to encourage
# more variables to be nonzero (gives more rounding flexibility)
# Use: minimize -s + epsilon * sum(vars)

for eps in [0.001, 0.01, 0.1, 0.0001]:
    c_obj = [eps]*12 + [-1]
    A_ub = [
        [-coeff_x[i] for i in range(12)] + [1],
        [-coeff_y[i] for i in range(12)] + [1],
        [-coeff_z[i] for i in range(12)] + [1],
        [-coeff_w[i] for i in range(12)] + [1],
    ]
    b_ub = [-9700, 22400, -3600, -150000]
    bounds = [(0, None)]*12 + [(None, None)]

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not result.success:
        continue

    exact = result.x[:12]
    s_val = result.x[12]
    nonzero_idx = [i for i in range(12) if exact[i] > 0.005]

    print(f"\neps={eps}, s={s_val:.2f}, nonzero: {[(names[i], round(exact[i],4)) for i in nonzero_idx]}")

    if len(nonzero_idx) > 12:
        continue

    # Try rounding
    floors = [max(0, np.floor(exact[i] * 100) / 100) for i in range(12)]
    ceils = [np.ceil(exact[i] * 100) / 100 for i in range(12)]

    if len(nonzero_idx) <= 16:
        best = None
        count = 0
        for combo in product([0, 1], repeat=len(nonzero_idx)):
            candidate = list(floors)
            for j, idx in enumerate(nonzero_idx):
                if combo[j] == 1:
                    candidate[idx] = ceils[idx]

            if check(candidate):
                s = sum(candidate)
                if best is None or s < best[0]:
                    best = (s, list(candidate))
                count += 1

        if best:
            print(f"=== FOUND {count} feasible roundings ===")
            show(best[1])
            break
        else:
            print("  No feasible rounding")

# Also try: manually construct solution
# K has huge X coefficient (+1125436). Even K=0.01 gives X += 11254.
# But K=-2074564 in Y, so K=0.01 gives Y -= 20745.
# With K=0.01: X += 11254, Y -= 20745, Z -= 28745, W -= 28745
# Need to compensate Y, Z, W with other variables

print("\n\n=== Manual construction approach ===")
# Use K to satisfy X, then use F to compensate Y, Z, W
# K=0.01: X += 11254.36, Y -= 20745.64, Z -= 28745.64, W -= 28745.64
# F=0.01: X -= 2499.498, Y += 4340.502, Z += 7760.502, W += 7760.502

# Let's try K and combinations of other positive-in-YZW variables
# Try grid search over K and a few helper vars
for K_val in np.arange(0.01, 0.05, 0.01):
    for F_val in np.arange(0.0, 0.10, 0.01):
        for A_val in np.arange(0.0, 5.0, 0.01):
            for D_val in np.arange(0.0, 2.0, 0.1):
                vals = [A_val, 0, 0, D_val, 0, F_val, 0, 0, 0, 0, K_val, 0]
                if check(vals):
                    print("Found!")
                    show([round(v,2) for v in vals])
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        continue
    break
else:
    print("Grid search didn't find solution")

# Simpler: just use B and K and J
print("\n=== Trying B, G, J, K approach ===")
# From max-min: B≈1.19, G≈7.93, J≈0.094
# The issue is J: X coeff = +124107, Y coeff = -217893
# Rounding J from 0.0939 to 0.09 loses X = 124107*0.004 = 484 but gains Y = 217893*0.004 = 849
# Rounding J to 0.10 gains X = 124107*0.006 = 745 but loses Y = 217893*0.006 = 1307

# What if we use more variables? Add L (X:+8080, Y:-13120)
# Or I (X:+7892, Y:-13308)

# Let's try: fix J=0.09, then solve remaining with LP
from scipy.optimize import linprog as lp

# Fix J=0.09, solve for remaining
J_fixed = 0.09
# Adjust RHS for J contribution
rhs_adj_x = 9700 - coeff_x[9] * J_fixed  # need remaining to exceed this
rhs_adj_y = -22400 - coeff_y[9] * J_fixed
rhs_adj_z = 3600 - coeff_z[9] * J_fixed
rhs_adj_w = 150000 - coeff_w[9] * J_fixed

print(f"After J=0.09: need X_rest > {rhs_adj_x:.1f}, Y_rest > {rhs_adj_y:.1f}, Z_rest > {rhs_adj_z:.1f}, W_rest > {rhs_adj_w:.1f}")

# Remaining vars: 0-8, 10, 11 (skip index 9=J)
rem_idx = [0,1,2,3,4,5,6,7,8,10,11]
c_rem = [-c for i,c in enumerate(coeff_x) if i != 9]
A_rem = [
    [-coeff_x[i] for i in rem_idx],
    [-coeff_y[i] for i in rem_idx],
    [-coeff_z[i] for i in rem_idx],
    [-coeff_w[i] for i in rem_idx],
]
b_rem = [
    -(9700 - coeff_x[9]*J_fixed),
    (22400 + coeff_y[9]*J_fixed),  # -Y <= 22400 + Y_J
    -(3600 - coeff_z[9]*J_fixed),
    -(150000 - coeff_w[9]*J_fixed),
]

# Maximize min slack with perturbation
n_rem = len(rem_idx)
c_obj3 = [0.0001]*n_rem + [-1]
A_ub3 = []
for row in A_rem:
    A_ub3.append(row + [1])
b_ub3 = b_rem
bounds3 = [(0, None)]*n_rem + [(None, None)]

r3 = lp(c_obj3, A_ub=A_ub3, b_ub=b_ub3, bounds=bounds3, method='highs')
if r3.success:
    print(f"Slack = {r3.x[-1]:.2f}")
    for i, idx in enumerate(rem_idx):
        if r3.x[i] > 0.005:
            print(f"  {names[idx]} = {r3.x[i]:.6f}")

    # Now try rounding
    exact_full = [0.0]*12
    exact_full[9] = J_fixed
    for i, idx in enumerate(rem_idx):
        exact_full[idx] = r3.x[i]

    nonzero_idx2 = [i for i in range(12) if exact_full[i] > 0.005]
    floors2 = [max(0, np.floor(exact_full[i] * 100) / 100) for i in range(12)]
    ceils2 = [np.ceil(exact_full[i] * 100) / 100 for i in range(12)]

    print(f"\nNonzero count: {len(nonzero_idx2)}")
    if len(nonzero_idx2) <= 16:
        best2 = None
        for combo in product([0, 1], repeat=len(nonzero_idx2)):
            candidate = list(floors2)
            for j, idx in enumerate(nonzero_idx2):
                if combo[j] == 1:
                    candidate[idx] = ceils2[idx]
            if check(candidate):
                s = sum(candidate)
                if best2 is None or s < best2[0]:
                    best2 = (s, list(candidate))
        if best2:
            print("=== SOLUTION FOUND ===")
            show(best2[1])
        else:
            print("No feasible rounding")
