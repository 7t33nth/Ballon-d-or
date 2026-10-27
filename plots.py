r"""
Графики для сравнения Кейна и Мбаппе.

Все рисунки сохраняются в figures/ и возвращаются как пути — ноутбук
показывает их через IPython.display.Image.

Запуск:
    python plots.py
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from mplsoccer import Pitch, Radar

import metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "clean"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

KANE = metrics.PLAYER_COLORS["Harry Kane"]
MBAPPE = metrics.PLAYER_COLORS["Kylian Mbappe"]
TEXT = "#1F2328"
MUTED = "#6B7280"
GRID = "#D8DEE4"
BG = "#FFFFFF"
PANEL = "#F6F8FA"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": GRID,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.titleweight": "bold",
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
})

SOURCE_UNDERSTAT = "Источник: Understat, лига 2025/26"
SOURCE_FBREF = "Источник: FBref, все турниры до 30.06.2026"
SOURCE_BOTH = "Источники: Understat (лига 2025/26), FBref (все турниры до 30.06.2026)"


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.3, facecolor=BG)
    plt.close(fig)
    return path


def _note(fig: plt.Figure, text: str, y: float = 0.01) -> None:
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8, color=MUTED)


def plot_radar() -> Path:
    data = pd.read_csv(CLEAN / "metrics.csv").set_index("player")
    cols = ["NPG+A_p90_pct", "xA_p90_pct", "KP_p90_pct", "npxG_p90_pct",
            "Conversion_pct", "NPShots_p90_pct"]
    params = ["Гол+пас\n/90", "xA\n/90", "Ключевые\nпередачи /90",
              "npxG\n/90", "Конверсия\n%", "Удары\n/90"]
    kane = data.loc["Harry Kane", cols].tolist()
    mbappe = data.loc["Kylian Mbappe", cols].tolist()

    radar = Radar(
        params, [0] * len(params), [100] * len(params),
        num_rings=4, ring_width=1, center_circle_radius=1,
    )
    fig, ax = radar.setup_axis(figsize=(9, 9))
    radar.draw_circles(ax=ax, facecolor=PANEL, edgecolor=GRID, linewidth=1)
    radar.draw_radar_compare(
        kane, mbappe, ax=ax,
        kwargs_radar={"facecolor": KANE, "alpha": 0.28, "edgecolor": KANE, "linewidth": 2},
        kwargs_compare={"facecolor": MBAPPE, "alpha": 0.28, "edgecolor": MBAPPE, "linewidth": 2},
    )
    range_labels = radar.draw_range_labels(ax=ax, fontsize=9, color=MUTED)
    for label in range_labels:
        x_pos, y_pos = label.get_position()
        if not (abs(x_pos) < 1e-6 and y_pos > 0):
            label.set_visible(False)
    radar.draw_param_labels(ax=ax, fontsize=11, color=TEXT)

    handles = [
        Patch(facecolor=KANE, alpha=0.35, edgecolor=KANE, label="Кейн"),
        Patch(facecolor=MBAPPE, alpha=0.35, edgecolor=MBAPPE, label="Мбаппе"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.10),
              ncol=2, frameon=False, fontsize=11)
    fig.suptitle("Профиль в перцентилях", y=0.99, fontsize=17, fontweight="bold")
    fig.text(
        0.5, 0.945,
        "100 = лучший показатель среди 12 форвардов: 10 лучших бомбардиров сезона + Кейн и Мбаппе",
        ha="center", fontsize=9.5, color=MUTED,
    )
    _note(fig, SOURCE_UNDERSTAT, y=0.015)
    return _save(fig, "radar.png")


def plot_shot_maps() -> Path:
    players = [
        ("Harry Kane", "shots_harry_kane.csv", KANE),
        ("Kylian Mbappe", "shots_kylian_mbappe.csv", MBAPPE),
    ]
    pitch = Pitch(
        pitch_type="statsbomb", pitch_color=PANEL, line_color="#AAB2BD",
        linewidth=1.1, goal_type="box",
    )
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 8.6))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(top=0.85, bottom=0.06, left=0.02, right=0.98, wspace=0.06)

    for ax, (player, filename, color) in zip(axs, players):
        shots = pd.read_csv(ROOT / "data/raw/understat" / filename)
        goals = shots["result"] == "Goal"
        x = shots["X"] * 120
        y = shots["Y"] * 80
        size = 28 + 520 * shots["xG"].clip(lower=0.005)

        pitch.draw(ax=ax)
        ax.set_xlim(55, 123)
        ax.set_ylim(-1.5, 81.5)
        ax.set_aspect("equal")

        ax.scatter(x[~goals], y[~goals], s=size[~goals], c="#B8C0C8",
                   alpha=0.55, edgecolors="white", linewidths=0.4, zorder=2)
        ax.scatter(x[goals], y[goals], s=size[goals], c=color,
                   alpha=0.92, edgecolors="white", linewidths=0.6, zorder=3)

        total = len(shots)
        goals_n = int(goals.sum())
        xg = shots["xG"].sum()
        diff = goals_n - xg
        ax.set_title(player, fontsize=16, pad=10)
        ax.text(
            0.03, 0.965,
            f"Удары: {total}   ·   Голы: {goals_n}   ·   xG: {xg:.1f}   ·   "
            f"Голы − xG: {diff:+.1f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
            color=TEXT, bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                  edgecolor=GRID, alpha=0.9),
        )
        handles = [
            Patch(facecolor=color, label="Гол"),
            Patch(facecolor="#B8C0C8", label="Не гол"),
        ]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.01),
                  frameon=False, fontsize=9)

    fig.suptitle("Карта ударов: лига 2025/26", y=0.975, fontsize=18, fontweight="bold")
    fig.text(0.5, 0.925, "Размер круга — xG удара; атака слева направо",
             ha="center", fontsize=10, color=MUTED)
    _note(fig, SOURCE_UNDERSTAT, y=0.012)
    return _save(fig, "shot_maps.png")


def plot_score_breakdown() -> Path:
    scores = pd.read_csv(CLEAN / "scores.csv", index_col=0)
    blocks = [
        ("Result (20)", "Гол+пас", "#D71920"),
        ("Creation (15)", "Создание", "#F28E2B"),
        ("Finishing (10)", "Завершение", "#59A14F"),
        ("Volume (5)", "Объём", "#4E79A7"),
        ("Team (30)", "Команда", "#B07AA1"),
        ("FairPlay (20)", "Фейр-плей", "#76B7B2"),
    ]
    players = ["Harry Kane", "Kylian Mbappe"]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    fig.patch.set_facecolor(BG)

    for row, player in enumerate(players):
        left = 0.0
        for col, _, color in blocks:
            value = float(scores.loc[player, col])
            ax.barh(row, value, left=left, height=0.55, color=color,
                    edgecolor="white", linewidth=1.2, zorder=2)
            if value >= 3.5:
                ax.text(left + value / 2, row, f"{value:.1f}", ha="center",
                        va="center", fontsize=8.5, color="white", fontweight="bold")
            left += value
        total = float(scores.loc[player, "TOTAL (100)"])
        ax.text(left + 1.2, row, f"{total:.1f}", va="center", fontsize=14,
                fontweight="bold", color=TEXT)

    ax.set_yticks(range(len(players)))
    ax.set_yticklabels(["Кейн", "Мбаппе"], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Баллы из 100")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)

    handles = [Patch(facecolor=color, label=label) for _, label, color in blocks]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.18),
              ncol=6, frameon=False, fontsize=9)
    ax.set_title("Из чего складывается счёт", pad=42, fontsize=16)
    _note(fig, "Веса: статистика 50 · команда 30 · фейр-плей 20. " + SOURCE_BOTH, y=-0.02)
    return _save(fig, "score_breakdown.png")


def plot_team_points() -> Path:
    team = pd.read_csv(CLEAN / "team_points.csv").set_index("player")
    players = ["Harry Kane", "Kylian Mbappe"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor(BG)

    y = np.arange(len(players))
    trophies = team.loc[players, "trophies"].values
    awards = team.loc[players, "awards"].values
    ax.barh(y, trophies, height=0.5, color="#4E79A7", edgecolor="white",
            linewidth=1.2, label="Трофеи")
    ax.barh(y, awards, left=trophies, height=0.5, color="#F28E2B",
            edgecolor="white", linewidth=1.2, label="Индивидуальные награды")

    for i, player in enumerate(players):
        if trophies[i] > 0:
            ax.text(trophies[i] / 2, i, f"{trophies[i]:.0f}", ha="center",
                    va="center", color="white", fontweight="bold")
        if awards[i] > 0:
            ax.text(trophies[i] + awards[i] / 2, i, f"{awards[i]:.0f}", ha="center",
                    va="center", color="white", fontweight="bold")
        ax.text(trophies[i] + awards[i] + 0.4, i, f"{trophies[i] + awards[i]:.0f}",
                va="center", fontsize=13, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(["Кейн", "Мбаппе"], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, 15)
    ax.set_xlabel("Сырые очки достижений")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2,
              frameon=False, fontsize=9)
    ax.set_title("Трофеи и индивидуальные награды", pad=38, fontsize=16)
    _note(fig, "Карьерные рекорды не учитываются. " + SOURCE_BOTH, y=-0.04)
    return _save(fig, "team_points.png")


def plot_goals_by_competition() -> Path:
    by_comp = pd.read_csv(CLEAN / "all_comps_by_comp.csv")
    players = ["Harry Kane", "Kylian Mbappe"]
    pivot = (
        by_comp.pivot_table(index="player", columns="competition_group",
                            values="np_goals", aggfunc="sum", fill_value=0)
        .reindex(index=players, columns=metrics.COMP_ORDER, fill_value=0)
    )
    colors = {
        "Лига": "#1F4E9C",
        "Лига чемпионов": "#7B4FA0",
        "Кубки": "#F28E2B",
        "Суперкубки": "#F2CF5B",
        "Клубный ЧМ": "#59A14F",
        "Сборная": "#76B7B2",
        "Другое": "#B8C0C8",
    }
    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor(BG)

    y = np.arange(len(players))
    left = np.zeros(len(players))
    for comp in metrics.COMP_ORDER:
        values = pivot[comp].values
        if values.sum() == 0:
            continue
        ax.barh(y, values, left=left, height=0.5, color=colors[comp],
                edgecolor="white", linewidth=1.0, label=comp)
        for i, value in enumerate(values):
            if value >= 2:
                ax.text(left[i] + value / 2, i, f"{value:.0f}", ha="center",
                        va="center", fontsize=8.5,
                        color="white" if comp not in ("Суперкубки",) else TEXT,
                        fontweight="bold")
        left += values

    for i, player in enumerate(players):
        total = int(pivot.loc[player].sum())
        ax.text(left[i] + 0.8, i, f"{total}", va="center", fontsize=13,
                fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(["Кейн", "Мбаппе"], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.15)
    ax.set_xlabel("Голы без пенальти")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=4,
              frameon=False, fontsize=8.5, columnspacing=1.2,
              handlelength=1.4, handletextpad=0.5)
    ax.set_title("Голы без пенальти по турнирам", pad=52, fontsize=16)
    _note(
        fig,
        "FBref, все турниры до 30.06.2026; плей-офф ЧМ не учтён. "
        "С учётом пенальти: Кейн 70, Мбаппе 55.",
        y=-0.05,
    )
    return _save(fig, "goals_by_competition.png")


def plot_fair_play() -> Path:
    fp = pd.read_csv(CLEAN / "fair_play_scores.csv").set_index("player")
    players = ["Harry Kane", "Kylian Mbappe"]
    labels = ["Кейн", "Мбаппе"]
    colors = [KANE, MBAPPE]

    panels = [
        ("Fls_p90", "Фолы / 90", "Fls", "фолов"),
        ("CrdY_p90", "Жёлтые карточки / 90", "CrdY", "ЖК"),
        ("DefActions_p90", "Оборонительные действия / 90", "DefActions", "действий"),
        ("DirtyIndex", "DirtyIndex (меньше = чище)", None, None),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.2))
    fig.patch.set_facecolor(BG)
    axs = axs.ravel()

    for ax, (col, title, count_col, count_label) in zip(axs, panels):
        values = fp.loc[players, col].values.astype(float)
        bars = ax.bar(labels, values, color=colors, width=0.55,
                      edgecolor="white", linewidth=1.2)
        for i, (bar, value) in enumerate(zip(bars, values)):
            label = f"{value:.2f}"
            if count_col:
                count = int(fp.loc[players[i], count_col])
                label += f"\n({count} {count_label})"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.03,
                    label, ha="center", va="bottom", fontsize=10,
                    fontweight="bold")
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_ylim(0, max(values) * 1.35 if max(values) > 0 else 1)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Фейр-плей: фолы, карточки и оборона", y=1.01,
                 fontsize=17, fontweight="bold")
    fig.text(
        0.5, 0.955,
        "Мбаппе фолит реже, но получает больше ЖК и почти не идёт в отбор — "
        "формула «на оборонительное действие» это учитывает",
        ha="center", fontsize=9.5, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    _note(fig, SOURCE_FBREF, y=0.01)
    return _save(fig, "fair_play.png")


def plot_sensitivity_scenarios() -> Path:
    data = pd.read_csv(CLEAN / "scores_sensitivity.csv")
    data = data.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.patch.set_facecolor(BG)
    ax.hlines(y, data["mbappe"], data["kane"], color="#D5DAE0", linewidth=4, zorder=1)
    ax.scatter(data["kane"], y, color=KANE, s=90, zorder=3, label="Кейн",
               edgecolors="white", linewidths=1)
    ax.scatter(data["mbappe"], y, color=MBAPPE, s=90, zorder=3, label="Мбаппе",
               edgecolors="white", linewidths=1)

    for i, row in data.iterrows():
        ax.text(max(row["kane"], row["mbappe"]) + 1.0, i, f"{row['gap']:+.1f}",
                va="center", fontsize=9, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(data["scenario"], fontsize=10)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Итоговый счёт из 100")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.axvline(50, color="#C4CAD1", linewidth=1, linestyle="--", zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=2,
              frameon=False, fontsize=10)
    ax.set_title("Чувствительность итога к сценариям", pad=40, fontsize=16)
    _note(fig, "Число справа — разрыв в пользу лидера. " + SOURCE_BOTH, y=-0.02)
    return _save(fig, "sensitivity_scenarios.png")


def plot_sensitivity_heatmap() -> Path:
    grid = pd.read_csv(CLEAN / "sensitivity_grid_stats.csv")
    pivot = grid.pivot_table(index="result_w", columns="creation_w",
                             values="gap", aggfunc="first")
    pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
    values = pivot.values.astype(float)

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(BG)
    fig, ax = plt.subplots(figsize=(9.2, 7))
    fig.patch.set_facecolor(BG)
    im = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap,
                   vmin=np.nanmin(values), vmax=np.nanmax(values))
    ax.set_facecolor(BG)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:g}" for x in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{y:g}" for y in pivot.index], fontsize=8)
    ax.set_xlabel("Вес Creation (создание)")
    ax.set_ylabel("Вес Result (гол+пас)")

    # Базовая точка 20/15
    if 20 in pivot.index and 15 in pivot.columns:
        xi = list(pivot.columns).index(15)
        yi = list(pivot.index).index(20)
        ax.scatter(xi, yi, marker="*", s=260, color="#1F4E9C",
                   edgecolor="white", linewidth=1.2, zorder=5)
        ax.annotate("база\n20/15", (xi, yi), textcoords="offset points",
                    xytext=(10, 10), fontsize=9, color="#1F4E9C",
                    fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Разрыв в пользу Кейна, баллы", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Чувствительность к весам внутри статистики", y=0.995,
                 fontsize=16, fontweight="bold")
    fig.text(
        0.5, 0.905,
        "Завершение = 10, объём = 50 − Result − Creation − 10; команда = 30, фейр-плей = 20.\n"
        "Кейн выигрывает во всех 70 комбинациях",
        ha="center", fontsize=9, color=MUTED,
    )
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.09, right=0.98)
    _note(fig, "Цвет — итоговый разрыв; звёздочка — базовые веса. " + SOURCE_BOTH, y=-0.02)
    return _save(fig, "sensitivity_heatmap.png")


def plot_all() -> dict[str, Path]:
    paths = {
        "radar": plot_radar(),
        "shot_maps": plot_shot_maps(),
        "score_breakdown": plot_score_breakdown(),
        "team_points": plot_team_points(),
        "goals_by_competition": plot_goals_by_competition(),
        "fair_play": plot_fair_play(),
        "sensitivity_scenarios": plot_sensitivity_scenarios(),
        "sensitivity_heatmap": plot_sensitivity_heatmap(),
    }
    for name, path in paths.items():
        print(f"{name}: {path}")
    return paths


if __name__ == "__main__":
    plot_all()
