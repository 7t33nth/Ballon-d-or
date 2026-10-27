r"""
Система баллов 0–100 для сравнения Кейна и Мбаппе (критерии ЗМ).

Структура (как в исходном scores.py, но с непрерывной нормализацией):
- 50% индивидуальная статистика
    - Result (20): NPG+A на 90 (лига)
    - Creation (15): среднее перцентилей xA/90 и ключевых передач/90
    - Finishing (10): среднее перцентилей «голы − npxG» и конверсии
    - Volume (5): перцентиль ударов без пенальти на 90
- 30% командные достижения: трофеи + индивидуальные награды (доля от суммы)
- 20% фейр-плей: индекс «грязности» из исходной формулы (доля от суммы)

Перцентили считаются внутри группы из 12 форвардов: 10 пула + Кейн + Мбаппе.
Для командных достижений и фейр-плей пула нет, поэтому используется доля
от суммы двух игроков — это сохраняет величину разрыва и не превращает блок в 0/100.

Запуск:
    python scores.py

Результаты:
    data/clean/scores.csv
    data/clean/block_percentiles.csv
    data/clean/team_points.csv
    data/clean/team_details.csv
    data/clean/fair_play_scores.csv
    data/clean/scores_sensitivity.csv
    data/clean/sensitivity_grid_major.csv
    data/clean/sensitivity_grid_stats.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "clean"

PLAYERS = metrics.PLAYER_ORDER
LABELS = metrics.PLAYER_LABELS

WEIGHTS = {
    "Result": 20,
    "Creation": 15,
    "Finishing": 10,
    "Volume": 5,
    "Team": 30,
    "FairPlay": 20,
}
STATS_KEYS = ["Result", "Creation", "Finishing", "Volume"]
MAJOR_KEYS = ["Stats", "Team", "FairPlay"]

TROPHY_WEIGHTS = {
    "bundesliga": 3, "premier league": 3, "la liga": 3, "serie a": 3, "ligue 1": 3,
    "champions league": 3, "world cup": 3,
    "dfb-pokal": 2, "coppa italia": 2, "copa del rey": 2, "french cup": 2, "german cup": 2,
    "club world cup": 2, "supercopa": 1, "super cup": 1, "dfl-supercup": 1,
}


def _norm(series: pd.Series) -> pd.Series:
    """Старая min-max нормализация — только для сценария «как было»."""
    if series.max() == series.min():
        return pd.Series(50.0, index=series.index)
    return (series - series.min()) / (series.max() - series.min()) * 100


def team_points() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Трофеи + награды. Карьерные рекорды попадают в таблицу, но не в счёт."""
    trophies = pd.read_csv(ROOT / "data/raw/trophies.csv")
    awards = pd.read_csv(ROOT / "data/raw/awards.csv")

    details = []
    rows = []
    for player in PLAYERS:
        trophy_score = 0.0
        for _, row in trophies[trophies["Player"] == player].iterrows():
            comp = str(row["Competition"])
            result = str(row["Result"]).strip().lower()
            is_winner = result in ("winner", "champion", "won")
            points = 0.0
            if is_winner:
                for key, weight in TROPHY_WEIGHTS.items():
                    if key in comp.lower():
                        points = float(weight)
                        break
                else:
                    points = 1.0
            trophy_score += points
            details.append({
                "player": player,
                "type": "Трофей",
                "item": comp,
                "points": points,
                "counted": is_winner,
                "note": str(row["Result"]),
            })

        award_score = 0.0
        for _, row in awards[awards["Player"] == player].iterrows():
            award = str(row["Award"])
            detail = str(row["Detail"])
            award_l = award.lower()
            detail_l = detail.lower()
            career_record = (
                "all-time" in award_l
                or "all time" in award_l
                or "record" in award_l
                or "record" in detail_l
                or "рекорд" in award_l
                or "рекорд" in detail_l
            )
            if career_record:
                points = 0.0
                note = "карьерный рекорд — не учитываем"
            elif "golden boot" in award_l or "golden shoe" in award_l or "бутса" in award_l:
                points = 3.0
                note = detail
            elif "best player" in award_l or "лучший игрок" in award_l:
                points = 2.0
                note = detail
            elif "top scorer" in award_l or "бомбардир" in award_l:
                points = 2.0
                note = detail
            else:
                points = 1.0
                note = detail
            award_score += points
            details.append({
                "player": player,
                "type": "Награда",
                "item": award,
                "points": points,
                "counted": not career_record,
                "note": note,
            })

        rows.append({
            "player": player,
            "trophies": trophy_score,
            "awards": award_score,
            "total": trophy_score + award_score,
        })

    team = pd.DataFrame(rows).set_index("player")
    if team["total"].sum() == 0:
        team["Team_pct"] = 50.0
    else:
        team["Team_pct"] = 100 * team["total"] / team["total"].sum()
    team["Team_pct"] = team["Team_pct"].round(1)
    return team, pd.DataFrame(details)


def fair_play_scores() -> pd.DataFrame:
    fp = metrics.fair_play()
    fp = fp.copy()
    # 1 / (1 + DirtyIndex) — монотонное преобразование, чтобы «чистота» была > 0.
    fp["clean"] = 1 / (1 + fp["DirtyIndex"])
    fp["FairPlay_pct"] = (100 * fp["clean"] / fp["clean"].sum()).round(1)
    fp["FairPlay_points"] = (fp["FairPlay_pct"] / 100 * WEIGHTS["FairPlay"]).round(1)
    return fp


def build_blocks(group: pd.DataFrame, team: pd.DataFrame, fp: pd.DataFrame) -> pd.DataFrame:
    subjects = group.set_index("player").loc[PLAYERS]
    blocks = pd.DataFrame(index=PLAYERS)
    blocks["Result_pct"] = subjects["NPG+A_p90_pct"]
    blocks["Creation_pct"] = (subjects["xA_p90_pct"] + subjects["KP_p90_pct"]) / 2
    blocks["Finishing_pct"] = (subjects["Conversion_pct"] + subjects["Overperformance_pct"]) / 2
    blocks["Volume_pct"] = subjects["NPShots_p90_pct"]
    blocks["Team_pct"] = team.loc[PLAYERS, "Team_pct"]
    blocks["FairPlay_pct"] = fp.loc[PLAYERS, "FairPlay_pct"]
    return blocks.round(1)


def major_weights(stats: float, team: float, fp: float) -> dict[str, float]:
    """Разбивает вес статистики на подблоки в исходной пропорции 20/15/10/5."""
    raw = np.array([WEIGHTS[key] for key in STATS_KEYS], dtype=float)
    raw = raw / raw.sum() * stats
    weights = {key: float(value) for key, value in zip(STATS_KEYS, raw)}
    weights["Team"] = float(team)
    weights["FairPlay"] = float(fp)
    return weights


def score_from_blocks(blocks: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    weights = dict(WEIGHTS if weights is None else weights)
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Сумма весов должна быть больше нуля")
    weights = {key: value / total_weight * 100 for key, value in weights.items()}

    scores = pd.DataFrame(index=blocks.index)
    for key in STATS_KEYS:
        scores[key] = blocks[f"{key}_pct"] / 100 * weights[key]
    scores["Stats"] = scores[STATS_KEYS].sum(axis=1)
    scores["Team"] = blocks["Team_pct"] / 100 * weights["Team"]
    scores["FairPlay"] = blocks["FairPlay_pct"] / 100 * weights["FairPlay"]
    scores["Total"] = scores[["Stats", "Team", "FairPlay"]].sum(axis=1)
    return scores


def to_display(scores: pd.DataFrame) -> pd.DataFrame:
    display = pd.DataFrame(index=scores.index)
    display[f"Result ({WEIGHTS['Result']})"] = scores["Result"]
    display[f"Creation ({WEIGHTS['Creation']})"] = scores["Creation"]
    display[f"Finishing ({WEIGHTS['Finishing']})"] = scores["Finishing"]
    display[f"Volume ({WEIGHTS['Volume']})"] = scores["Volume"]
    display["Stats (50)"] = scores["Stats"]
    display[f"Team ({WEIGHTS['Team']})"] = scores["Team"]
    display[f"FairPlay ({WEIGHTS['FairPlay']})"] = scores["FairPlay"]
    display["TOTAL (100)"] = scores["Total"]
    return display.round(1)


def compute_base(pool_flag: str = "pool_goals") -> dict:
    group = metrics.build_group(pool_flag)
    team, details = team_points()
    fp = fair_play_scores()
    blocks = build_blocks(group, team, fp)
    scores = score_from_blocks(blocks)
    return {
        "pool_flag": pool_flag,
        "group": group,
        "team": team,
        "team_details": details,
        "fp": fp,
        "blocks": blocks,
        "scores": scores,
    }


def old_minmax_scores(base: dict) -> pd.DataFrame:
    """Воспроизводит старую логику: min-max по двум игрокам в каждом блоке."""
    group = base["group"].set_index("player").loc[PLAYERS]
    team = base["team"]
    fp = base["fp"]

    blocks = pd.DataFrame(index=PLAYERS)
    blocks["Result_pct"] = _norm(group["NPG+A_p90"])
    blocks["Creation_pct"] = _norm((group["xA_p90"] + group["KP_p90"]) / 2)
    blocks["Finishing_pct"] = (
        _norm(group["Overperformance"]) + _norm(group["Conversion"])
    ) / 2
    blocks["Volume_pct"] = _norm(group["NPShots_p90"])
    blocks["Team_pct"] = _norm(team["total"])
    blocks["FairPlay_pct"] = _norm(-fp["DirtyIndex"])
    return score_from_blocks(blocks)


def scenario_row(name: str, scores: pd.DataFrame, note: str) -> dict:
    kane = scores.loc["Harry Kane", "Total"]
    mbappe = scores.loc["Kylian Mbappe", "Total"]
    gap = kane - mbappe
    winner = "Кейн" if gap > 0 else "Мбаппе" if gap < 0 else "ничья"
    return {
        "scenario": name,
        "kane": round(kane, 1),
        "mbappe": round(mbappe, 1),
        "gap": round(gap, 1),
        "winner": winner,
        "note": note,
    }


def sensitivity_scenarios(base: dict) -> pd.DataFrame:
    rows = [scenario_row("База", base["scores"], "50/30/20, перцентили, основной пул")]

    rows.append(scenario_row(
        "Старый min-max",
        old_minmax_scores(base),
        "как в исходном scores.py: каждый блок 0/100",
    ))

    for name, weights, note in [
        ("Только статистика", major_weights(100, 0, 0), "веса перенормированы до 100"),
        ("Только команда", major_weights(0, 100, 0), "веса перенормированы до 100"),
        ("Только фейр-плей", major_weights(0, 0, 100), "веса перенормированы до 100"),
        ("Равные блоки", major_weights(100 / 3, 100 / 3, 100 / 3), "33/33/33"),
        ("Без фейр-плея", major_weights(55, 45, 0), "55/45/0"),
        ("Без команды", major_weights(70, 0, 30), "70/0/30"),
    ]:
        rows.append(scenario_row(name, score_from_blocks(base["blocks"], weights), note))

    alt = compute_base("pool_ga90")
    rows.append(scenario_row(
        "Альтернативный пул",
        alt["scores"],
        "топ-10 по G+A/90 вместо топ-10 по голам",
    ))

    team_no_dfl = base["team"].copy()
    team_no_dfl.loc["Harry Kane", "trophies"] -= 1
    team_no_dfl["total"] = team_no_dfl["trophies"] + team_no_dfl["awards"]
    team_no_dfl["Team_pct"] = (100 * team_no_dfl["total"] / team_no_dfl["total"].sum()).round(1)
    blocks_no_dfl = base["blocks"].copy()
    blocks_no_dfl["Team_pct"] = team_no_dfl["Team_pct"]
    rows.append(scenario_row(
        "Без DFL-Supercup",
        score_from_blocks(blocks_no_dfl),
        "у Кейна на один трофей меньше",
    ))

    team_with_record = base["team"].copy()
    team_with_record.loc["Kylian Mbappe", "awards"] += 2
    team_with_record["total"] = team_with_record["trophies"] + team_with_record["awards"]
    team_with_record["Team_pct"] = (
        100 * team_with_record["total"] / team_with_record["total"].sum()
    ).round(1)
    blocks_with_record = base["blocks"].copy()
    blocks_with_record["Team_pct"] = team_with_record["Team_pct"]
    rows.append(scenario_row(
        "С карьерным рекордом",
        score_from_blocks(blocks_with_record),
        "Мбаппе +2 за «All-time top scorer»",
    ))

    team_trophies_only = base["team"].copy()
    team_trophies_only["awards"] = 0
    team_trophies_only["total"] = team_trophies_only["trophies"]
    team_trophies_only["Team_pct"] = (
        100 * team_trophies_only["total"] / team_trophies_only["total"].sum()
    ).round(1)
    blocks_trophies_only = base["blocks"].copy()
    blocks_trophies_only["Team_pct"] = team_trophies_only["Team_pct"]
    rows.append(scenario_row(
        "Только трофеи",
        score_from_blocks(blocks_trophies_only),
        "индивидуальные награды не входят в командный блок",
    ))

    return pd.DataFrame(rows)


def weight_grid_major(base: dict) -> pd.DataFrame:
    rows = []
    for stats in np.arange(40, 60.1, 2.5):
        for team in np.arange(20, 40.1, 2.5):
            fp = 100 - stats - team
            if fp < 0:
                continue
            scores = score_from_blocks(base["blocks"], major_weights(stats, team, fp))
            kane = scores.loc["Harry Kane", "Total"]
            mbappe = scores.loc["Kylian Mbappe", "Total"]
            rows.append({
                "stats_w": round(stats, 1),
                "team_w": round(team, 1),
                "fairplay_w": round(fp, 1),
                "kane": round(kane, 1),
                "mbappe": round(mbappe, 1),
                "gap": round(kane - mbappe, 1),
                "winner": "Кейн" if kane > mbappe else "Мбаппе" if mbappe > kane else "ничья",
            })
    return pd.DataFrame(rows)


def weight_grid_stats(base: dict) -> pd.DataFrame:
    rows = []
    for result in np.arange(5, 35.1, 2.5):
        for creation in np.arange(5, 35.1, 2.5):
            finishing = 10.0
            volume = 50.0 - result - creation - finishing
            if volume < 0 or volume > 15:
                continue
            weights = {
                "Result": result,
                "Creation": creation,
                "Finishing": finishing,
                "Volume": volume,
                "Team": 30,
                "FairPlay": 20,
            }
            scores = score_from_blocks(base["blocks"], weights)
            kane = scores.loc["Harry Kane", "Total"]
            mbappe = scores.loc["Kylian Mbappe", "Total"]
            rows.append({
                "result_w": round(result, 1),
                "creation_w": round(creation, 1),
                "finishing_w": round(finishing, 1),
                "volume_w": round(volume, 1),
                "kane": round(kane, 1),
                "mbappe": round(mbappe, 1),
                "gap": round(kane - mbappe, 1),
                "winner": "Кейн" if kane > mbappe else "Мбаппе" if mbappe > kane else "ничья",
            })
    return pd.DataFrame(rows)


def main() -> None:
    base = compute_base("pool_goals")
    display = to_display(base["scores"])
    blocks = base["blocks"].round(1)
    team = base["team"].round(1)
    fp = base["fp"].round(2)

    display.to_csv(CLEAN / "scores.csv", encoding="utf-8-sig")
    blocks.to_csv(CLEAN / "block_percentiles.csv", encoding="utf-8-sig")
    team.to_csv(CLEAN / "team_points.csv", encoding="utf-8-sig")
    base["team_details"].to_csv(CLEAN / "team_details.csv", index=False, encoding="utf-8-sig")
    fp.to_csv(CLEAN / "fair_play_scores.csv", encoding="utf-8-sig")

    scenarios = sensitivity_scenarios(base)
    scenarios.to_csv(CLEAN / "scores_sensitivity.csv", index=False, encoding="utf-8-sig")

    grid_major = weight_grid_major(base)
    grid_major.to_csv(CLEAN / "sensitivity_grid_major.csv", index=False, encoding="utf-8-sig")

    grid_stats = weight_grid_stats(base)
    grid_stats.to_csv(CLEAN / "sensitivity_grid_stats.csv", index=False, encoding="utf-8-sig")

    print("=== БЛОКИ, 0–100 ===")
    print(blocks.to_string())
    print("\n=== БАЛЛЫ ===")
    print(display.to_string())
    print("\n=== КОМАНДНЫЕ ДОСТИЖЕНИЯ ===")
    print(team.to_string())
    print("\n=== ФЕЙР-ПЛЕЙ ===")
    print(fp[["CrdY", "CrdR", "Fls", "DefActions", "Fls_per_DA",
              "CrdY_per_DA", "DirtyIndex", "FairPlay_pct"]].to_string())
    print("\n=== СЦЕНАРИИ ===")
    print(scenarios.to_string(index=False))

    print("\n=== ЧУВСТВИТЕЛЬНОСТЬ ===")
    print(f"Основная сетка: разрыв от {grid_major['gap'].min():.1f} до "
          f"{grid_major['gap'].max():.1f}; победители: {grid_major['winner'].unique().tolist()}")
    stats_wins = grid_stats[grid_stats["winner"] == "Мбаппе"]
    print(f"Сетка внутри статистики: разрыв от {grid_stats['gap'].min():.1f} до "
          f"{grid_stats['gap'].max():.1f}; комбинаций с победой Мбаппе: "
          f"{len(stats_wins)} из {len(grid_stats)}")
    if len(stats_wins):
        best = stats_wins.sort_values("gap").iloc[0]
        print(f"  максимальный перевес Мбаппе: {abs(best['gap']):.1f} при "
              f"Result={best['result_w']}, Creation={best['creation_w']}, "
              f"Finishing={best['finishing_w']}, Volume={best['volume_w']}")

    print(f"\nСохранено в {CLEAN}")


if __name__ == "__main__":
    main()
