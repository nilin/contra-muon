from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
X_MIN = 1e-3
NS_ITERATE = 8
POWER_BASIS_SIZE = NS_ITERATE
POWER_FIT_EPS = 1e-3
POWER_UNDERFIT_SAFETY = 1 - 1e-6
Y_MAX = 3.0

P_MINUS02_CONSTANT = 3.242828118701584
P_MINUS02_COEFFS = np.array([
    -0.2,
    -0.24246845196737132,
    -0.18980953526334665,
    -0.27850572679265884,
    -0.21520433117007404,
    -0.3353448658536533,
    -0.46222931635067455,
    -0.3192658913038056,
    0.0,
    0.0,
    0.0,
    0.0,
])


def phi_5(x: np.ndarray) -> np.ndarray:
    return 2 * x - 1.5 * x**3 + 0.5 * x**5


def iterate_phi(x: np.ndarray, n: int) -> np.ndarray:
    y = x.copy()
    for _ in range(n):
        y = phi_5(y)
    return y


def power_basis(x: np.ndarray, count: int = POWER_BASIS_SIZE) -> list[np.ndarray]:
    terms = [x.copy()]
    y = x.copy()
    for _ in range(count):
        y = phi_5(y)
        terms.append(y.copy())
    return terms


def power_minus02_ns_fit(x: np.ndarray) -> np.ndarray:
    # Use the TEST_MUON.py p=-0.2 coefficients, truncated so the top term is phi^8.
    terms = power_basis(x, POWER_BASIS_SIZE)
    coeffs = P_MINUS02_COEFFS[:POWER_BASIS_SIZE]
    fit = P_MINUS02_CONSTANT * terms[-1]
    for coeff, term in zip(coeffs, terms[:-1]):
        fit = fit + coeff * term
    return fit / (P_MINUS02_CONSTANT + coeffs.sum())


def fit_power_coefficients_under(p: float) -> np.ndarray:
    x_fit = np.unique(np.concatenate([
        np.geomspace(POWER_FIT_EPS, 1.0, 6000),
        np.linspace(POWER_FIT_EPS, 1.0, 2000),
    ]))
    B = np.stack(power_basis(x_fit, POWER_BASIS_SIZE), axis=1)
    y = x_fit**p

    result = linprog(
        c=-B.mean(axis=0),
        A_ub=B,
        b_ub=y,
        bounds=[(0.0, None)] * B.shape[1],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return POWER_UNDERFIT_SAFETY * result.x


P_PLUS02_COEFFS = fit_power_coefficients_under(0.2)
POWER_REFERENCES = (
    (r"Actual $x^{-0.2}$", -0.2),
    (r"Actual $x^{0.2}$", 0.2),
)


def fitted_power_ns(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return np.stack(power_basis(x, coeffs.size - 1), axis=1) @ coeffs


def p_minus02_coefficients_for_phi8() -> np.ndarray:
    coeffs = np.empty(POWER_BASIS_SIZE + 1)
    coeffs[:-1] = P_MINUS02_COEFFS[:POWER_BASIS_SIZE]
    coeffs[-1] = P_MINUS02_CONSTANT
    return coeffs / coeffs.sum()


def cumulative_from_high_iterates(x: np.ndarray, coeffs: np.ndarray) -> list[tuple[int, np.ndarray]]:
    terms = power_basis(x, coeffs.size - 1)
    total = np.zeros_like(x)
    curves = []
    for index in range(coeffs.size - 1, -1, -1):
        total = total + coeffs[index] * terms[index]
        curves.append((index, total.copy()))
    return curves


def curves(x: np.ndarray) -> dict[str, np.ndarray]:
    f = iterate_phi(x, NS_ITERATE)
    return {
        r"SGD $x$": x,
        rf"Muon $f=\phi^{{{NS_ITERATE}}}$": f,
        r"Contra $2f-x$": 2 * f - x,
        r"Power fit $p=-0.2$": power_minus02_ns_fit(x),
        r"Power fit $p=0.2$": fitted_power_ns(x, P_PLUS02_COEFFS),
    }


def style_axis(ax, *, logx: bool, ylabel: str, ylim: tuple[float, float]) -> None:
    if logx:
        ax.set_xscale("log")
        ax.set_xlim(X_MIN, 1.0)
    else:
        ax.set_xlim(0.0, 1.0)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"normalized singular value $r=s_i/s_1$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22)


def plot_maps(output: Path, *, logx: bool) -> None:
    x = np.geomspace(X_MIN, 1.0, 1600) if logx else np.linspace(X_MIN, 1.0, 1600)
    ys = curves(x)

    figsize = (6.0, 6.0) if not logx else (7.2, 4.5)
    fig, ax = plt.subplots(figsize=figsize, dpi=240)
    colors = {
        r"SGD $x$": "0.62",
        rf"Muon $f=\phi^{{{NS_ITERATE}}}$": "0.25",
        r"Contra $2f-x$": "#0072b2",
        r"Power fit $p=-0.2$": "#d55e00",
        r"Power fit $p=0.2$": "#009e73",
    }
    linestyles = {
        r"SGD $x$": "-",
        rf"Muon $f=\phi^{{{NS_ITERATE}}}$": "--",
        r"Contra $2f-x$": "-",
        r"Power fit $p=-0.2$": "-",
        r"Power fit $p=0.2$": "-",
    }
    for label, y in ys.items():
        ax.plot(x, y, color=colors[label], linestyle=linestyles[label], linewidth=2.1, label=label)
    for label, p in POWER_REFERENCES:
        ax.plot(x, x**p, color="0.0", linestyle=":", linewidth=1.0, label=label)
    ax.axhline(1, color="0.55", linestyle=":", linewidth=1.0)
    style_axis(ax, logx=logx, ylabel=r"update singular value map $a(r)$", ylim=(0.0, Y_MAX))
    ax.set_title("Muon, Contra-Muon, and Power-Muon singular-value maps")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if logx:
        fig.savefig(output, bbox_inches="tight")
    else:
        fig.savefig(output)
    plt.close(fig)
    print(output)


def plot_readme_figure(output: Path) -> None:
    x = np.geomspace(X_MIN, 1.0, 1800)
    ys = curves(x)
    colors = {
        r"SGD $x$": "0.62",
        rf"Muon $f=\phi^{{{NS_ITERATE}}}$": "0.25",
        r"Contra $2f-x$": "#0072b2",
        r"Power fit $p=-0.2$": "#d55e00",
        r"Power fit $p=0.2$": "#009e73",
    }
    linestyles = {
        r"SGD $x$": "-",
        rf"Muon $f=\phi^{{{NS_ITERATE}}}$": "--",
        r"Contra $2f-x$": "-",
        r"Power fit $p=-0.2$": "-",
        r"Power fit $p=0.2$": "-",
    }

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 7.2), dpi=240, sharex=True)
    for label, y in ys.items():
        ax0.plot(x, y, color=colors[label], linestyle=linestyles[label], linewidth=2.0, label=label)
        ax1.plot(x, x * y, color=colors[label], linestyle=linestyles[label], linewidth=2.0, label=label)
    for label, p in POWER_REFERENCES:
        y = x**p
        ax0.plot(x, y, color="0.0", linestyle=":", linewidth=1.0, label=label)
        ax1.plot(x, x * y, color="0.0", linestyle=":", linewidth=1.0, label=label)

    ax0.axhline(1, color="0.55", linestyle=":", linewidth=1.0)
    ax1.axhline(1, color="0.55", linestyle=":", linewidth=1.0)
    style_axis(ax0, logx=True, ylabel=r"update map $a(r)$", ylim=(0.0, Y_MAX))
    style_axis(ax1, logx=True, ylabel=r"first-order contribution $r\,a(r)$", ylim=(0.0, 1.08))
    ax0.set_title("Update singular values")
    ax1.set_title("Approximate loss-change contribution")
    ax0.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("How Muon variants reweight singular directions", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(output)


def plot_cumulative_power_fits(output: Path, *, logx: bool) -> None:
    x = np.geomspace(X_MIN, 1.0, 1800) if logx else np.linspace(X_MIN, 1.0, 1800)
    configs = (
        (-0.2, p_minus02_coefficients_for_phi8(), "#d55e00"),
        (0.2, P_PLUS02_COEFFS, "#009e73"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), dpi=240, sharex=True, sharey=True)
    for ax, (p, coeffs, color) in zip(axes, configs):
        cumulative = cumulative_from_high_iterates(x, coeffs)
        shades = np.linspace(0.25, 0.9, len(cumulative))
        for step, (index, y) in enumerate(cumulative):
            is_final = step == len(cumulative) - 1
            label = "final fit" if is_final else rf"through $\phi^{{{index}}}$"
            ax.plot(
                x,
                y,
                color=color,
                alpha=1.0 if is_final else shades[step],
                linewidth=2.2 if is_final else 1.0,
                label=label,
            )
        ax.plot(x, x**p, color="0.0", linestyle=":", linewidth=1.1, label=rf"actual $x^{{{p}}}$")
        ax.axhline(1, color="0.55", linestyle=":", linewidth=1.0)
        style_axis(ax, logx=logx, ylabel=r"cumulative update map $a(r)$", ylim=(0.0, Y_MAX))
        ax.set_title(rf"Cumulative fit for $p={p}$")
        ax.legend(frameon=False, fontsize=7, loc="upper left")

    fig.suptitle("Power-Muon cumulative linear combinations from highest iterate downward", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(output)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_maps(FIGURES / "power_muon_maps.png", logx=False)
    plot_readme_figure(FIGURES / "power_muon_readme_figure.png")
    plot_cumulative_power_fits(FIGURES / "power_muon_cumulative_fits_linear.png", logx=False)


if __name__ == "__main__":
    main()
