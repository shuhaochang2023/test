import numpy as np
from scipy.optimize import linprog
from itertools import product

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
    return X > 9950 and Y > -22300 and W > 3600 and Z > 150000

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
    print(f"  Y = {Y:.4f}  (need > -22300)  {'OK' if Y > -22300 else 'FAIL'}")
    print(f"  W = {W:.4f}  (need > 3600)    {'OK' if W > 3600 else 'FAIL'}")
    print(f"  Z = {Z:.4f}  (need > 150000)  {'OK' if Z > 150000 else 'FAIL'}")

# Constraints: X>9950, Y>-22300, W>3600, Z>150000
A_ub = [
    [-c for c in coeff_x],
    [-c for c in coeff_y],
    [-c for c in coeff_w],
    [-c for c in coeff_z],
]
b_ub = [-9950, 22300, -3600, -150000]
bounds = [(0, None)] * 12

# Step 1: Check continuous feasibility
print("=== Checking continuous feasibility ===")
result = linprog([1]*12, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
if not result.success:
    print("INFEASIBLE in continuous case")
    # Quick relaxation check
    coeffs_all = [coeff_x, coeff_y, coeff_z, coeff_w]
    targets = [9950, -22300, 150000, 3600]
    tnames = ['X', 'Y', 'Z', 'W']
    for i in range(4):
        others = [j for j in range(4) if j != i]
        A_sub = []
        b_sub = []
        for j in others:
            A_sub.append([-c for c in coeffs_all[j]])
            if tnames[j] == 'Y':
                b_sub.append(-targets[j])  # -Y <= 22300
            else:
                b_sub.append(-targets[j])
        # Fix b_sub for Y
        b_sub_fixed = []
        for j in others:
            if tnames[j] == 'Y':
                b_sub_fixed.append(22300)
            else:
                b_sub_fixed.append(-targets[j])

        c_max = [-c for c in coeffs_all[i]]  # maximize = minimize negative
        r = linprog(c_max, A_ub=A_sub, b_ub=b_sub_fixed, bounds=bounds, method='highs')
        if r.success:
            best_val = sum(coeffs_all[i][k]*r.x[k] for k in range(12))
            print(f"  Max {tnames[i]} with others satisfied: {best_val:.2f} (need {'> ' + str(targets[i])})")
    import sys; sys.exit(0)

print("Continuous solution EXISTS!")
exact = result.x
for i, (name, v) in enumerate(zip(names, exact)):
    if v > 0.0001:
        print(f"  {name} = {v:.6f}")

# Step 2: Maximize slack
print("\n=== Maximizing min slack ===")
c_obj = [0]*12 + [-1]
A_ub2 = [
    [-coeff_x[i] for i in range(12)] + [1],
    [-coeff_y[i] for i in range(12)] + [1],
    [-coeff_w[i] for i in range(12)] + [1],
    [-coeff_z[i] for i in range(12)] + [1],
]
b_ub2 = [-9950, 22300, -3600, -150000]
bounds2 = [(0, None)]*12 + [(None, None)]

r2 = linprog(c_obj, A_ub=A_ub2, b_ub=b_ub2, bounds=bounds2, method='highs')
if r2.success:
    print(f"Max min-slack = {r2.x[12]:.4f}")
    exact2 = r2.x[:12]
    nonzero = [(i, exact2[i]) for i in range(12) if exact2[i] > 0.005]
    for i, v in nonzero:
        print(f"  {names[i]} = {v:.6f}")

    X = calc(coeff_x, exact2)
    Y = calc(coeff_y, exact2)
    Z = calc(coeff_z, exact2)
    W = calc(coeff_w, exact2)
    print(f"  X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}, W={W:.2f}")

# Step 3: Try rounding with various margins
print("\n=== Finding rounded solution ===")
for margin in [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
    b_ub_m = [-(9950+margin), (22300-margin), -(3600+margin), -(150000+margin)]
    r3 = linprog([1]*12, A_ub=A_ub, b_ub=b_ub_m, bounds=bounds, method='highs')
    if not r3.success:
        print(f"  margin={margin}: infeasible")
        continue

    exact3 = r3.x
    nz3 = [i for i in range(12) if exact3[i] > 0.005]

    floors = [max(0, np.floor(exact3[i] * 100) / 100) for i in range(12)]
    ceils = [np.ceil(exact3[i] * 100) / 100 for i in range(12)]

    if len(nz3) <= 18:
        best = None
        for combo in product([0, 1], repeat=len(nz3)):
            candidate = list(floors)
            for j, idx in enumerate(nz3):
                if combo[j] == 1:
                    candidate[idx] = ceils[idx]
            if check(candidate):
                s = sum(candidate)
                if best is None or s < best[0]:
                    best = (s, list(candidate))
        if best:
            print(f"\n=== SOLUTION FOUND (margin={margin}) ===")
            show(best[1])
            break
        else:
            print(f"  margin={margin}: {len(nz3)} vars, no rounding works")
    else:
        print(f"  margin={margin}: {len(nz3)} vars (too many for exhaustive)")

# Step 4: If no solution yet, try fixing some vars and solving
else:
    print("\n=== Fix-and-solve approach ===")
    # Try different combinations of fixed J values
    for J_val in np.arange(0.08, 0.20, 0.01):
        J_val = round(J_val, 2)
        for L_val in np.arange(0.0, 0.50, 0.01):
            L_val = round(L_val, 2)
            rem_idx = [0,1,2,3,4,5,6,7,8,10]  # skip J(9) and L(11)
            n_rem = len(rem_idx)

            fixed_contrib = [coeff_x[9]*J_val + coeff_x[11]*L_val,
                           coeff_y[9]*J_val + coeff_y[11]*L_val,
                           coeff_w[9]*J_val + coeff_w[11]*L_val,
                           coeff_z[9]*J_val + coeff_z[11]*L_val]

            A_sub = []
            for row_coeffs in [coeff_x, coeff_y, coeff_w, coeff_z]:
                A_sub.append([-row_coeffs[i] for i in rem_idx])

            b_sub = [-(9950 - fixed_contrib[0]),
                     (22300 + fixed_contrib[1]),
                     -(3600 - fixed_contrib[2]),
                     -(150000 - fixed_contrib[3])]

            r4 = linprog([1]*n_rem, A_ub=A_sub, b_ub=b_sub, bounds=[(0,None)]*n_rem, method='highs')
            if not r4.success:
                continue

            exact_full = [0.0]*12
            exact_full[9] = J_val
            exact_full[11] = L_val
            for i, idx in enumerate(rem_idx):
                exact_full[idx] = r4.x[i]

            nz = [i for i in range(12) if exact_full[i] > 0.005]
            floors = [max(0, np.floor(exact_full[i]*100)/100) for i in range(12)]
            ceils = [np.ceil(exact_full[i]*100)/100 for i in range(12)]

            if len(nz) <= 16:
                for combo in product([0,1], repeat=len(nz)):
                    candidate = list(floors)
                    for j, idx in enumerate(nz):
                        if combo[j] == 1:
                            candidate[idx] = ceils[idx]
                    if check(candidate):
                        print(f"\n=== SOLUTION FOUND (J={J_val}, L={L_val}) ===")
                        show(candidate)
                        import sys; sys.exit(0)

    print("No solution found with fix-and-solve either.")
