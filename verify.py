#!/usr/bin/env python3
"""Verify the solution A~L for the linear system."""

# All variables
A, B, C, D, E, F, G, H, I, J, K, L = 0, 1.18, 0, 0, 0, 0, 8.00, 0, 0.06, 0.09, 0, 0

X = -8160*A - 62.4*B - 6750*C - 6943.1*D - 118219.5*E - 249949.8*F - 214.6*G - 7727.4*H + 7892*I + 124107*J + 1125436*K + 8080*L
Y = 13040*A - 62.4*B - 6750*C - 6943.1*D + 223780.5*E + 434050.2*F - 214.6*G + 13472.6*H - 13308*I - 217893*J - 2074564*K - 13120*L
Z = 34240*A - 62.4*B + 164250*C + 164057*D + 223780.5*E + 776050.2*F + 5085.4*G + 24072.6*H - 29208*I - 388893*J - 2874564*K - 34320*L
W = 55440*A + 56937.6*B + 335250*C + 506057*D + 52780.5*E + 776050.2*F + 15685.4*G + 24072.6*H - 39808*I - 445893*J - 2874564*K - 55520*L

print("=== Solution ===")
print(f"  A = {A}")
print(f"  B = {B}")
print(f"  C = {C}")
print(f"  D = {D}")
print(f"  E = {E}")
print(f"  F = {F}")
print(f"  G = {G}")
print(f"  H = {H}")
print(f"  I = {I}")
print(f"  J = {J}")
print(f"  K = {K}")
print(f"  L = {L}")
print()
print(f"  X = {X:.4f}  (need > 9700)    {'✓' if X > 9700 else '✗'}")
print(f"  Y = {Y:.4f}  (need > -22400)  {'✓' if Y > -22400 else '✗'}")
print(f"  Z = {Z:.4f}  (need > 3600)    {'✓' if Z > 3600 else '✗'}")
print(f"  W = {W:.4f}  (need > 150000)  {'✓' if W > 150000 else '✗'}")
