import numpy as np
from scipy.optimize import linprog
from itertools import product

# Coefficients (same equations)
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

# Constraints in <= form:
# -X <= -9950
# -Y <= 22200
# -W <= -3600  (W > 3600)
# -Z <= -150000  (Z > 150000)

# Max-min slack approach: maximize s
# -X + s <= -9950
# -Y + s <= 22200
# -W + s <= -3600
# -Z + s <= -150000

# Try with different epsilon to encourage more nonzero vars
for eps in [0, 0.0001, 0.001, 0.01, 0.1]:
    c_obj = [eps]*12 + [-1]
    A_ub = [
        [-coeff_x[i] for i in range(12)] + [1],
        [-coeff_y[i] for i in range(12)] + [1],
        [-coeff_w[i] for i in range(12)] + [1],  # W > 3600
        [-coeff_z[i] for i in range(12)] + [1],  # Z > 150000
    ]
    b_ub = [-9950, 22200, -3600, -150000]
    bounds = [(0, None)]*12 + [(None, None)]

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not result.success:
        print(f"eps={eps}: infeasible")
        continue

    exact = result.x[:12]
    s_val = result.x[12]
    nonzero_idx = [i for i in range(12) if exact[i] > 0.005]

    print(f"eps={eps}, slack={s_val:.2f}, nonzero: {[(names[i], round(exact[i],4)) for i in nonzero_idx]}")

    # Try rounding
    floors = [max(0, np.floor(exact[i] * 100) / 100) for i in range(12)]
    ceils = [np.ceil(exact[i] * 100) / 100 for i in range(12)]

    if len(nonzero_idx) <= 16:
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
            print("=== SOLUTION FOUND ===")
            show(best[1])
            break
        else:
            print("  No feasible rounding")

else:
    # Try fixing variables and solving sub-problems
    print("\n=== Trying fix-and-solve approach ===")

    # Z > 150000 is the hard one. Z has large positive coefficients for:
    # A(+34240), C(+164250), D(+164057), E(+223780.5), F(+776050.2), G(+5085.4), H(+24072.6)
    # And large negative for: I(-29208), J(-388893), K(-2874564), L(-34320)

    # X > 9950 needs: I(+7892), J(+124107), K(+1125436), L(+8080)
    # These are all negative in Z!

    # So we need a balance. Let's try fixing J at various values
    for J_fixed in np.arange(0.08, 0.15, 0.01):
        J_fixed = round(J_fixed, 2)
        rem_idx = [0,1,2,3,4,5,6,7,8,10,11]
        n_rem = len(rem_idx)

        c_obj2 = [0.0001]*n_rem + [-1]
        A_ub2 = [
            [-coeff_x[i] for i in rem_idx] + [1],
            [-coeff_y[i] for i in rem_idx] + [1],
            [-coeff_w[i] for i in rem_idx] + [1],
            [-coeff_z[i] for i in rem_idx] + [1],
        ]
        b_ub2 = [
            -(9950 - coeff_x[9]*J_fixed),
            (22200 + coeff_y[9]*J_fixed),
            -(3600 - coeff_w[9]*J_fixed),
            -(150000 - coeff_z[9]*J_fixed),
        ]
        bounds2 = [(0, None)]*n_rem + [(None, None)]

        r2 = linprog(c_obj2, A_ub=A_ub2, b_ub=b_ub2, bounds=bounds2, method='highs')
        if not r2.success:
            continue

        exact_full = [0.0]*12
        exact_full[9] = J_fixed
        for i, idx in enumerate(rem_idx):
            exact_full[idx] = r2.x[i]

        nonzero_idx2 = [i for i in range(12) if exact_full[i] > 0.005]
        floors2 = [max(0, np.floor(exact_full[i] * 100) / 100) for i in range(12)]
        ceils2 = [np.ceil(exact_full[i] * 100) / 100 for i in range(12)]

        print(f"J={J_fixed}, slack={r2.x[-1]:.2f}, nonzero: {[(names[i], round(exact_full[i],4)) for i in nonzero_idx2]}")

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
                break
            else:
                print("  No feasible rounding")
    else:
        # Try two fixed variables
        print("\n=== Trying two fixed variables ===")
        for J_fixed in [0.09, 0.10, 0.11, 0.12]:
            for I_fixed in [0.0, 0.01, 0.02, 0.03, 0.05]:
                rem_idx = [0,1,2,3,4,5,6,7,10,11]  # skip I(8) and J(9)
                n_rem = len(rem_idx)

                fixed_x = coeff_x[8]*I_fixed + coeff_x[9]*J_fixed
                fixed_y = coeff_y[8]*I_fixed + coeff_y[9]*J_fixed
                fixed_w = coeff_w[8]*I_fixed + coeff_w[9]*J_fixed
                fixed_z = coeff_z[8]*I_fixed + coeff_z[9]*J_fixed

                c_obj3 = [0.0001]*n_rem + [-1]
                A_ub3 = [
                    [-coeff_x[i] for i in rem_idx] + [1],
                    [-coeff_y[i] for i in rem_idx] + [1],
                    [-coeff_w[i] for i in rem_idx] + [1],
                    [-coeff_z[i] for i in rem_idx] + [1],
                ]
                b_ub3 = [
                    -(9950 - fixed_x),
                    (22200 + fixed_y),
                    -(3600 - fixed_w),
                    -(150000 - fixed_z),
                ]
                bounds3 = [(0, None)]*n_rem + [(None, None)]

                r3 = linprog(c_obj3, A_ub=A_ub3, b_ub=b_ub3, bounds=bounds3, method='highs')
                if not r3.success:
                    continue

                exact_full = [0.0]*12
                exact_full[8] = I_fixed
                exact_full[9] = J_fixed
                for i, idx in enumerate(rem_idx):
                    exact_full[idx] = r3.x[i]

                nonzero_idx3 = [i for i in range(12) if exact_full[i] > 0.005]
                floors3 = [max(0, np.floor(exact_full[i] * 100) / 100) for i in range(12)]
                ceils3 = [np.ceil(exact_full[i] * 100) / 100 for i in range(12)]

                if len(nonzero_idx3) <= 16:
                    best3 = None
                    for combo in product([0, 1], repeat=len(nonzero_idx3)):
                        candidate = list(floors3)
                        for k, idx in enumerate(nonzero_idx3):
                            if combo[k] == 1:
                                candidate[idx] = ceils3[idx]
                        if check(candidate):
                            s = sum(candidate)
                            if best3 is None or s < best3[0]:
                                best3 = (s, list(candidate))
                    if best3:
                        print(f"J={J_fixed}, I={I_fixed}")
                        print("=== SOLUTION FOUND ===")
                        show(best3[1])
                        import sys; sys.exit(0)
