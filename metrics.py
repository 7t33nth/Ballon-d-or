r"""
Метрики для сравнения Кейна и Мбаппе.

Что делает:
1. Собирает таблицу «10 форвардов пула + Кейн + Мбаппе» на данных Understat (лига 2025/26).
2. Считает per-90 метрики и перцентили внутри этой группы из 12 игроков.
3. Считает контекст по всем турнирам из FBref (голы, ассисты, минуты, дисциплина).
4. Готовит данные по фейр-плей (фолы и карточки на оборонительное действие).

Результаты:
    data/clean/metrics.csv             — основной пул (топ-10 по голам)
    data/clean/metrics_altpool.csv     — альтернативный пул (топ-10 по G+A/90)
    data/clean/all_comps.csv           — сводка по всем турнирам (FBref)
    data/clean/all_comps_by_comp.csv   — разбивка по турнирам
    data/clean/fair_play.csv           — дисциплина и индекс «грязности»

Запуск:
    python metrics.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent

PLAYER_ORDER = ["Harry Kane", "Kylian Mbappe"]
PLAYER_LABELS = {"Harry Kane": "Кейн", "Kylian Mbappe": "Мбаппе"}
PLAYER_COLORS = {"Harry Kane": "#D71920", "Kylian Mbappe": "#1F4E9C"}
PLAYER_META = {
    "Harry Kane": {"league": "Bundesliga", "team": "Bayern Munich"},
    "Kylian Mbappe": {"league": "La liga", "team": "Real Madrid"},
}
FBREF_PATHS = {
    "Harry Kane": ROOT / "data/raw/Kane/KaneFBRef.csv",
    "Kylian Mbappe": ROOT / "data/raw/Mbappe/MbappeFBRef.csv",
}

POOL_FLAGS = ("pool_goals", "pool_ga90")
POOL_LABELS = {
    "pool_goals": "топ-10 по голам",
    "pool_ga90": "топ-10 по G+A/90",
}

# Метрики, по которым считаются перцентили
METRIC_COLS = {
    "NPG+A_p90": "Гол+пас без пенальти /90",
    "xA_p90": "xA /90",
    "KP_p90": "Ключевые передачи /90",
    "npxG_p90": "npxG /90",
    "NPShots_p90": "Удары без пенальти /90",
    "Conversion": "Конверсия, %",
    "Overperformance": "Голы − npxG",
}

FBREF_NUMERIC = [
    "Min", "Gls", "Ast", "PK", "PKatt", "Sh", "SoT",
    "CrdY", "CrdR", "Fls", "Fld", "Off", "TklW", "Int",
]

COMP_ORDER = [
    "Лига",
    "Лига чемпионов",
    "Кубки",
    "Суперкубки",
    "Клубный ЧМ",
    "Сборная",
    "Другое",
]


def load_league_summary() -> pd.DataFrame:
    path = ROOT / "data/clean/summary_all.csv"
    summary = pd.read_csv(path).set_index("Player")
    missing = [player for player in PLAYER_ORDER if player not in summary.index]
    if missing:
        raise ValueError(f"В {path} нет игроков: {missing}. Сначала запусти load_clean.py.")
    return summary


def load_pool(pool_flag: str = "pool_goals") -> pd.DataFrame:
    if pool_flag not in POOL_FLAGS:
        raise ValueError(f"pool_flag должен быть одним из {POOL_FLAGS}")
    path = ROOT / "data/raw/understat/pool_candidates.csv"
    pool = pd.read_csv(path)
    return pool[pool[pool_flag]].copy()


def build_group(pool_flag: str = "pool_goals") -> pd.DataFrame:
    """Таблица из 12 игроков: 10 пула + Кейн + Мбаппе, с per-90 и перцентилями."""
    summary = load_league_summary()
    pool = load_pool(pool_flag)

    rows = []
    for player in PLAYER_ORDER:
        row = summary.loc[player]
        rows.append({
            "player": player,
            "group": PLAYER_LABELS[player],
            "is_subject": True,
            "league": PLAYER_META[player]["league"],
            "team": PLAYER_META[player]["team"],
            "minutes": row["Min"],
            "goals": row["Goals"],
            "assists": row["Assists"],
            "npg": row["NPGoals"],
            "npxG": row["npxG"],
            "xA": row["xA"],
            "key_passes": row["KeyPasses"],
            "shots": row["Shots"],
            "NPShots": row["NPShots"],
        })

    for _, row in pool.iterrows():
        rows.append({
            "player": row["player_name"],
            "group": "Пул",
            "is_subject": False,
            "league": row["league"],
            "team": row["team_title"],
            "minutes": row["time"],
            "goals": row["goals"],
            "assists": row["assists"],
            "npg": row["npg"],
            "npxG": row["npxG"],
            "xA": row["xA"],
            "key_passes": row["key_passes"],
            "shots": row["shots"],
            "NPShots": row["NPShots"],
        })

    group = pd.DataFrame(rows)
    group["NPG+A_p90"] = (group["npg"] + group["assists"]) / group["minutes"] * 90
    group["xA_p90"] = group["xA"] / group["minutes"] * 90
    group["KP_p90"] = group["key_passes"] / group["minutes"] * 90
    group["npxG_p90"] = group["npxG"] / group["minutes"] * 90
    group["NPShots_p90"] = group["NPShots"] / group["minutes"] * 90
    group["Conversion"] = group["npg"] / group["NPShots"] * 100
    group["Overperformance"] = group["npg"] - group["npxG"]

    # Перцентиль внутри группы из 12 игроков (10 пула + 2 субъекта).
    for col in METRIC_COLS:
        group[col + "_pct"] = group[col].apply(
            lambda value: percentileofscore(group[col], value, kind="rank")
        )

    return group


def load_fbref(path: Path) -> pd.DataFrame:
    """FBref-выгрузка: двухстрочная шапка + итоговая строка в конце."""
    df = pd.read_csv(path, encoding="utf-8-sig", skiprows=1)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()  # убираем итоговую строку
    for col in FBREF_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["Min"].fillna(0) > 0].copy()
    return df


def all_comps_summary() -> pd.DataFrame:
    rows = []
    for player, path in FBREF_PATHS.items():
        df = load_fbref(path)
        minutes = int(df["Min"].sum())
        goals = int(df["Gls"].sum())
        pen_goals = int(df["PK"].sum())
        assists = int(df["Ast"].sum())
        np_goals = goals - pen_goals
        shots = int(df["Sh"].sum())
        rows.append({
            "player": player,
            "matches": len(df),
            "minutes": minutes,
            "goals": goals,
            "pen_goals": pen_goals,
            "np_goals": np_goals,
            "assists": assists,
            "np_g_a": np_goals + assists,
            "np_g_a_p90": round((np_goals + assists) / minutes * 90, 2),
            "goals_p90": round(goals / minutes * 90, 2),
            "assists_p90": round(assists / minutes * 90, 2),
            "shots": shots,
            "shots_p90": round(shots / minutes * 90, 2),
            "crdy": int(df["CrdY"].sum()),
            "crdr": int(df["CrdR"].sum()),
            "fouls": int(df["Fls"].sum()),
            "tackles_won": int(df["TklW"].sum()),
            "interceptions": int(df["Int"].sum()),
            "first_match": df["Date"].min().date(),
            "last_match": df["Date"].max().date(),
        })
    return pd.DataFrame(rows).set_index("player")


def competition_group(comp: str) -> str:
    comp = str(comp)
    if comp in {"Bundesliga", "La Liga", "EPL", "Serie A", "Ligue 1"}:
        return "Лига"
    if comp == "Champions Lg":
        return "Лига чемпионов"
    if comp in {"DFB-Pokal", "Copa del Rey", "FA Cup", "Coupe de France", "Coppa Italia"}:
        return "Кубки"
    if comp in {"DFL-Supercup", "Supercopa de España", "Super Cup"}:
        return "Суперкубки"
    if comp == "Club World Cup":
        return "Клубный ЧМ"
    if comp in {"World Cup", "WCQ", "Friendlies (M)", "UEFA Nations League"}:
        return "Сборная"
    return "Другое"


def all_comps_by_comp() -> pd.DataFrame:
    frames = []
    for player, path in FBREF_PATHS.items():
        df = load_fbref(path)
        df["competition_group"] = df["Comp"].map(competition_group)
        grouped = (
            df.groupby("competition_group", as_index=False)[
                ["Min", "Gls", "Ast", "PK", "Sh", "CrdY", "CrdR", "Fls", "TklW", "Int"]
            ]
            .sum()
        )
        grouped["player"] = player
        grouped["np_goals"] = grouped["Gls"] - grouped["PK"]
        frames.append(grouped)
    result = pd.concat(frames, ignore_index=True)
    result["competition_group"] = pd.Categorical(
        result["competition_group"], categories=COMP_ORDER, ordered=True
    )
    return result.sort_values(["player", "competition_group"]).reset_index(drop=True)


def fair_play() -> pd.DataFrame:
    rows = []
    for player, path in FBREF_PATHS.items():
        df = load_fbref(path)
        minutes = int(df["Min"].sum())
        crdy = int(df["CrdY"].sum())
        crdr = int(df["CrdR"].sum())
        fouls = int(df["Fls"].sum())
        tackles = int(df["TklW"].sum())
        interceptions = int(df["Int"].sum())
        def_actions = tackles + interceptions

        fls_per_da = fouls / def_actions if def_actions else np.nan
        crdy_per_da = crdy / def_actions if def_actions else np.nan
        rows.append({
            "player": player,
            "minutes": minutes,
            "CrdY": crdy,
            "CrdR": crdr,
            "Fls": fouls,
            "TklW": tackles,
            "Int": interceptions,
            "DefActions": def_actions,
            "Fls_p90": round(fouls / minutes * 90, 2),
            "CrdY_p90": round(crdy / minutes * 90, 2),
            "DefActions_p90": round(def_actions / minutes * 90, 2),
            "Fls_per_DA": round(fls_per_da, 3),
            "CrdY_per_DA": round(crdy_per_da, 3),
            # Формула из исходного scores.py: карточки весим выше фолов.
            "DirtyIndex": round(3 * crdy_per_da + fls_per_da, 3),
        })
    return pd.DataFrame(rows).set_index("player")


def save_all() -> None:
    clean = ROOT / "data/clean"
    clean.mkdir(parents=True, exist_ok=True)

    main = build_group("pool_goals")
    alt = build_group("pool_ga90")
    main.to_csv(clean / "metrics.csv", index=False, encoding="utf-8-sig")
    alt.to_csv(clean / "metrics_altpool.csv", index=False, encoding="utf-8-sig")

    all_comps = all_comps_summary()
    all_comps.to_csv(clean / "all_comps.csv", encoding="utf-8-sig")

    by_comp = all_comps_by_comp()
    by_comp.to_csv(clean / "all_comps_by_comp.csv", index=False, encoding="utf-8-sig")

    fp = fair_play()
    fp.to_csv(clean / "fair_play.csv", encoding="utf-8-sig")

    print("=== Индивидуальные метрики (лига, основной пул) ===")
    cols = ["player", "minutes", "npg", "assists", "NPG+A_p90", "xA_p90", "KP_p90",
            "npxG_p90", "NPShots_p90", "Conversion", "Overperformance"]
    print(main[cols].round(2).to_string(index=False))
    print("\n=== Перцентили (0–100, группа из 12) ===")
    pct_cols = ["player"] + [col + "_pct" for col in METRIC_COLS]
    print(main[pct_cols].round(1).to_string(index=False))
    print("\n=== Все турниры (FBref, до 30.06.2026) ===")
    print(all_comps[["matches", "minutes", "goals", "np_goals", "assists",
                     "np_g_a_p90", "shots", "last_match"]].to_string())
    print("\n=== Фейр-плей ===")
    print(fp[["CrdY", "CrdR", "Fls", "DefActions", "Fls_per_DA",
              "CrdY_per_DA", "DirtyIndex"]].to_string())
    print(f"\nСохранено в {clean}")


if __name__ == "__main__":
    save_all()
