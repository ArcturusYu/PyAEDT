from re import M
from sympy import pi, tan, symbols, I, cos, sin, Matrix, sqrt, oo, re, im, Abs

# Define the symbols
c, f, Z_0, l_4 = symbols('c f Z_0 l_4')
beta = symbols(r'\beta', real=True)
Z_07_factor = symbols('Z_07_factor', real=True)
Z_0 = 1
Z_07 = Z_0 / sqrt(2)
l_stub_factor = 1
l_stub = l_4 * l_stub_factor
freq_factor = symbols('f', real=True)
Z_stub = Z_0

M1 = Matrix([[1, 0],
             [I * tan(1*pi*freq_factor*l_stub_factor) / Z_stub, 1]])

M2 = Matrix([[cos(0.5*pi*freq_factor), I * Z_0 * sin(0.5*pi*freq_factor)],
             [I / Z_0 * sin(0.5*pi*freq_factor), cos(0.5*pi*freq_factor)]])

Meven = Matrix([[1, 0],
                [I * tan(0.25*pi*freq_factor) / Z_0, 1]])

Modd = Matrix([[1, 0],
               [-I * Z_0 / tan(0.25*pi*freq_factor), 1]])

M3 = Matrix([[cos(0.5*pi*freq_factor), I * Z_07 * sin(0.5*pi*freq_factor)],
             [I / Z_07 * sin(0.5*pi*freq_factor), cos(0.5*pi*freq_factor)]])

M3_ = Matrix([[cos(0.25*pi*freq_factor), I * Z_07 * sin(0.25*pi*freq_factor)],
              [I / Z_07 * sin(0.25*pi*freq_factor), cos(0.25*pi*freq_factor)]])

M_e = Matrix([[1, 0],
              [0, 1]])

M_o = Matrix([[1, 0],
              [oo, 1]])

def a2gamma(amatrix):
    a = amatrix[0, 0]
    b = amatrix[0, 1]
    c = amatrix[1, 0]
    d = amatrix[1, 1]
    z = Z_0
    gamma = (a+b/z-c*z-d)/(a+b/z+c*z+d)
    return gamma

def a2tau(amatrix):
    a = amatrix[0, 0]
    b = amatrix[0, 1]
    c = amatrix[1, 0]
    d = amatrix[1, 1]
    z = Z_0
    tau = 2/(a+b/z+c*z+d)
    return tau

a_e_ = M1 * M2 * Meven * M3 * Meven * M2 * M1
a_o_ = M1 * M2 * Modd * M3 * Modd * M2 * M1
gamma_e_ = a2gamma(a_e_)
gamma_o_ = a2gamma(a_o_)
tau_e_ = a2tau(a_e_)
tau_o_ = a2tau(a_o_)
s11_ = 0.5 * (gamma_e_ + gamma_o_)
s12_ = 0.5 * (tau_e_ + tau_o_)
s13_ = 0.5 * (tau_e_ - tau_o_)
s14_ = 0.5 * (gamma_e_ - gamma_o_)

a_e = Meven * M3 * Meven
a_o = Modd * M3 * Modd
gamma_e = a2gamma(a_e)
gamma_o = a2gamma(a_o)
tau_e = a2tau(a_e)
tau_o = a2tau(a_o)
s11 = 0.5 * (gamma_e + gamma_o)
s12 = 0.5 * (tau_e + tau_o)
s13 = 0.5 * (tau_e - tau_o)
s14 = 0.5 * (gamma_e - gamma_o)

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def normalize_angles(angles):
    normalized_angles = [(angle + 180) % 360 - 180 for angle in angles]
    return normalized_angles

# Create numerical functions
# s11_func = sp.lambdify(freq_factor, s11, modules='numpy')
# s12_func = sp.lambdify(freq_factor, s12, modules='numpy')
# s13_func = sp.lambdify(freq_factor, s13, modules='numpy')
# s14_func = sp.lambdify(freq_factor, s14, modules='numpy')
s11_func_ = sp.lambdify(freq_factor, s11_, modules='numpy')
s12_func_ = sp.lambdify(freq_factor, s12_, modules='numpy')
s13_func_ = sp.lambdify(freq_factor, s13_, modules='numpy')
s14_func_ = sp.lambdify(freq_factor, s14_, modules='numpy')


# Generate freq_factor values
freq_values = np.linspace(0.9, 1.1, 1000)

# Evaluate functions
# s11_values = s11_func(freq_values)
# s12_values = s12_func(freq_values)
# s13_values = s13_func(freq_values)
# s14_values = s14_func(freq_values)
s11_values_ = s11_func_(freq_values)
s12_values_ = s12_func_(freq_values)
s13_values_ = s13_func_(freq_values)
s14_values_ = s14_func_(freq_values)

# Compute magnitudes in dB
# mag_s11_dB = 20 * np.log10(np.abs(s11_values))
# mag_s12_dB = 20 * np.log10(np.abs(s12_values))
# mag_s13_dB = 20 * np.log10(np.abs(s13_values))
# mag_s14_dB = 20 * np.log10(np.abs(s14_values))
mag_s11_dB_ = 20 * np.log10(np.abs(s11_values_))
mag_s12_dB_ = 20 * np.log10(np.abs(s12_values_))
mag_s13_dB_ = 20 * np.log10(np.abs(s13_values_))
mag_s14_dB_ = 20 * np.log10(np.abs(s14_values_))

# Compute angles in degrees

# angle_s12_deg = np.angle(s12_values, deg=True)
# angle_s13_deg = np.angle(s13_values, deg=True)
# ang_s12_minus_s13_deg = angle_s12_deg - angle_s13_deg
angle_s12_deg_ = np.angle(s12_values_, deg=True)
angle_s13_deg_ = np.angle(s13_values_, deg=True)
ang_s12_minus_s13_deg_ = angle_s12_deg_ - angle_s13_deg_

# ang_s12_minus_s13_deg = normalize_angles(ang_s12_minus_s13_deg)
ang_s12_minus_s13_deg_ = normalize_angles(ang_s12_minus_s13_deg_)

# Plot magnitudes in dB
plt.figure(figsize=(10, 6))

# 定义目标dB值
target_dB = -20

# 定义一个函数来找到交点
# def find_intersection(freq_values, mag_values, target_dB):
#     intersections = []
#     for i in range(len(mag_values) - 1):
#         if (mag_values[i] - target_dB) * (mag_values[i + 1] - target_dB) < 0:
#             # 线性插值计算交点
#             x1, x2 = freq_values[i], freq_values[i + 1]
#             y1, y2 = mag_values[i], mag_values[i + 1]
#             x_intersect = x1 + (target_dB - y1) * (x2 - x1) / (y2 - y1)
#             intersections.append(x_intersect)
#     return intersections

# # 找到所有曲线的交点
# intersections_s11 = find_intersection(freq_values, mag_s11_dB, target_dB)
# intersections_s12 = find_intersection(freq_values, mag_s12_dB, target_dB)
# intersections_s13 = find_intersection(freq_values, mag_s13_dB, target_dB)
# intersections_s14 = find_intersection(freq_values, mag_s14_dB, target_dB)
# intersections_s11_ = find_intersection(freq_values, mag_s11_dB_, target_dB)
# intersections_s12_ = find_intersection(freq_values, mag_s12_dB_, target_dB)
# intersections_s13_ = find_intersection(freq_values, mag_s13_dB_, target_dB)
# intersections_s14_ = find_intersection(freq_values, mag_s14_dB_, target_dB)

# plt.plot(freq_values, mag_s11_dB, label='|S11| (dB)', linestyle='--')
# plt.plot(freq_values, mag_s12_dB, label='|S12| (dB)', linestyle='--')
# plt.plot(freq_values, mag_s13_dB, label='|S13| (dB)', linestyle='--')
# plt.plot(freq_values, mag_s14_dB, label='|S14| (dB)', linestyle='--')
# plt.plot(freq_values, mag_s11_dB_, label='|S11_| (dB)')
# plt.plot(freq_values, mag_s12_dB_, label='|S12_| (dB)')
# plt.plot(freq_values, mag_s13_dB_, label='|S13_| (dB)')
# plt.plot(freq_values, mag_s14_dB_, label='|S14_| (dB)')

plt.title('Magnitude of S-parameters vs. freq_factor')
plt.xlabel('freq_factor')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)
plt.show()

# Plot angles in degrees
plt.figure(figsize=(10, 6))

plt.plot(freq_values, ang_s12_minus_s13_deg_, label='∠S12 - ∠S13 (deg)')

plt.title('Phase Angles of S-parameters vs. freq_factor')
plt.xlabel('freq_factor')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.grid(True)
plt.show()


# Define the range for l_stub_factor
l_stub_factor_values = np.linspace(0.95, 105, 11)

# Modify M1 matrix to include l_stub
for l_stub_factor_value in l_stub_factor_values:

    # Recompute M1 with the new l_stub_factor
    M1 = Matrix([[1, 0],
                 [I * tan(1*pi*l_stub_factor_value) / Z_stub, 1]])

    # Recompute the a_e_, a_o_, gamma_e_, and gamma_o_
    a_e_ = M1 * M2 * Meven * M3 * Meven * M2 * M1
    a_o_ = M1 * M2 * Modd * M3 * Modd * M2 * M1

    # Recompute gamma_e_ and gamma_o_
    gamma_e_ = a2gamma(a_e_)
    gamma_o_ = a2gamma(a_o_)

    # Calculate S11 as the average of gamma_e_ and gamma_o_
    s11_ = 0.5 * (gamma_e_ + gamma_o_)

    # Create a numerical function for the new S11
    s11_func_ = sp.lambdify(freq_factor, s11_, modules='numpy')

    # Evaluate the function over the frequency range
    s11_values_ = s11_func_(freq_values)

    # Compute magnitude in dB
    mag_s11_dB_ = 20 * np.log10(np.abs(s11_values_))

    # Plot the magnitude in dB for the current l_stub_factor
    plt.plot(freq_values, mag_s11_dB_, label=f'|S11| (dB) for l_stub_factor={l_stub_factor_value:.2f}')

# Add plot details
plt.title('Magnitude of S11 vs. freq_factor for different l_stub_factor values')
plt.xlabel('freq_factor')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)
plt.show()