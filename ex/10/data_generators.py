"""Class 10, Exercises A, B: Equation Discovery

This module contains data generator functions for exercise class 10. It was
written, presumably, by Jure Brence, and was slightly modified by me.

"""

import pandas as pd
import numpy as np


def generate_newton(n_cases, noise=0):
    """Generate data for F = m a"""
    np.random.seed(1234)
    data = {"F": [], "m": [], "a": []}
    for _ in range(n_cases):
        m, a = np.random.rand(2)
        f = m * a * (1 + np.random.normal(0, noise))
        data["F"].append(f)
        data["m"].append(m)
        data["a"].append(a)
    return pd.DataFrame(data)


def generate_stefan(n_cases, noise=0):
    """Generate data for j = sigma T^4"""
    np.random.seed(1234)
    data = {"j": [], "T": []}
    sigma = 5.67 * 10 ** -8
    for _ in range(n_cases):
        t = 100 * np.random.rand(1)[0] + 100
        j = sigma * t ** 4 * (1 + np.random.normal(0, noise))
        data["j"].append(j)
        data["T"].append(t)
    return pd.DataFrame(data)


def generate_lorentz(n_cases, noise=0):
    """Generate data for gamma = sqrt(1 - (v / c)^2)"""
    np.random.seed(1234)
    data = {"gamma": [], "v": []}
    c = 3 * 10 ** 5  # [km / s]
    for _ in range(n_cases):
        v = np.random.rand(1)[0] * c
        gamma = np.sqrt(1 - (v / c) ** 2) * (1 + np.random.normal(0, noise))
        data["gamma"].append(gamma)
        data["v"].append(v)
    return pd.DataFrame(data)


def generate_conservation_of_energy_const(n_cases, noise=0):
    """Generate data for m g h + 0.5 m v^2 = c.

    We assume all bodies have the same initial energy c.
    """
    np.random.seed(1234)
    data = {"m": [], "h": [], "v": []}
    c = 100  # [J]
    g = 9.81  # [m/s^2]
    for _ in range(n_cases):
        m, h = np.random.rand(2)
        h = h * 10  # med 0 in 10 metri --> 0 <= Wp = m g h < 1 * 10 * 10 = 100
        v = np.sqrt((c - m * g * h) * 2 / m) * (1 + np.random.normal(0, noise))
        data["m"].append(m)
        data["v"].append(v)
        data["h"].append(h)
    return pd.DataFrame(data)


def generate_conservation_of_energy(n_cases, noise=0):
    """Generate data for m g h + 0.5 m v^2 = c.

    We assume all bodies have the same initial energy c.
    """
    np.random.seed(1234)
    data = {"E": [], "m": [], "h": [], "v": []}
    g = 9.81  # [m/s^2]
    for _ in range(n_cases):
        m, h, v = np.random.rand(3)
        #h = h * 10  # med 0 in 10 metri 
        #v = v * 10 # med 0 in 10 metri na sekundo
        data["m"].append(m)
        data["v"].append(v)
        data["h"].append(h)
        data["E"].append((m*g*h + 0.5*m*v**2)*(1 + np.random.normal(0, noise)))
    return pd.DataFrame(data)


def generate_surface(n_cases, noise=0):
    """Generate data for p = n a ** 2 / (4 tan (pi / n))"""
    np.random.seed(1234)
    data = {"n": [], "a": [], "S": []}
    for n in range(3, n_cases + 3):
        a = np.random.rand(1)[0]
        s = n * a ** 2 / (4 * np.tan(np.pi / n)) * (1 + np.random.normal(0, noise))
        data["n"].append(n)
        data["a"].append(a)
        data["S"].append(s)
    return pd.DataFrame(data)


def generate_linear(n_cases, noise=0):
    """Generate data for y = x1 + 3x2"""
    np.random.seed(1234)
    data = {"y": [], "x1": [], "x2": []}
    for i in range(n_cases):
        x1, x2 = np.random.rand(2)
        y = (x1 + 3*x2) * (1 + np.random.normal(0, noise))
        data["y"].append(y)
        data["x1"].append(x1)
        data["x2"].append(x2)
    return pd.DataFrame(data)


def generirate_circular_motion(n_cases, noise=0):
    """Generate data for y = r sin(1.337 pi t)"""
    np.random.seed(1234)
    data = {"y": [], "r": [], "t": []}
    for i in range(n_cases):
        r, t = np.random.rand(2)
        y =  r*np.sin(1.337*np.pi*t) * (1 + np.random.normal(0, noise))
        data["y"].append(y)
        data["r"].append(r)
        data["t"].append(t)
    return pd.DataFrame(data)
