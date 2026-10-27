r"""
Очистка выгрузок Understat (CSV с разделителем ";").
Ожидает папку с четырьмя файлами одного игрока:
    PositionX.csv, ShotTypesX.csv, ShotZonesX.csv, SituationX.csv
(имена могут быть любыми, таблицы определяются по названию первой колонки).
Запуск из корня проекта (Windows):
    python load_clean.py data\raw\Mbappe --player "Kylian Mbappe"
Что делает:
1. Читает файлы (разделитель ";"), определяет таблицу по первой колонке.
2. Убирает служебный столбец "number" и приводит названия колонок к единому виду.
3. Проверяет, что итоги разных таблиц совпадают между собой (удары, голы, xG).
4. Считает сводку игрока: голы без пенальти, npxG, значения на 90 минут.
5. Сохраняет чистые таблицы в data/clean/<игрок>/ и обновляет data/clean/summary_all.csv.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EXPECTED = ["position", "shotTypes", "shotZones", "situation"]
# Как называется первая (категориальная) колонка в каждой таблице после очистки
CATEGORY_NAME = {
    "position": "Position",
    "shotTypes": "ShotType",
    "shotZones": "ShotZone",
    "situation": "Situation",
}
# Единые названия остальных колонок
RENAME = {
    "apps": "Apps", "min": "Min", "goals": "Goals", "assists": "Assists",
    "sp90m": "Shots_p90", "kp90": "KeyPasses_p90",
    "shots": "Shots", "kp": "KeyPasses",
    "xG": "xG", "xA": "xA", "xG90": "xG_p90", "xA90": "xA_p90",
    "xGperSh": "xG_per_shot", "xAperKP": "xA_per_keypass",
}
OUT_NAMES = {
    "position": "position.csv",
    "shotTypes": "shot_types.csv",
    "shotZones": "shot_zones.csv",
    "situation": "situation.csv",
}
def read_table(path: Path) -> tuple[str, pd.DataFrame]:
    """Читает один файл Understat и возвращает (тип таблицы, чистый DataFrame)."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig").dropna(how="all")
    df = df.drop(columns=["number"], errors="ignore")
    key = df.columns[0]
    if key not in EXPECTED:
        raise ValueError(f"первая колонка '{key}', ожидал одну из {EXPECTED}")
    category = CATEGORY_NAME[key]
    df = df.rename(columns={key: category, **RENAME})
    for col in df.columns:
        if col != category:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return key, df
def load_player(folder: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(folder.glob("*.csv")):
        # В той же папке лежат выгрузки FBref — их обрабатывает metrics.py.
        if "fbref" in path.name.lower():
            continue
        try:
            key, df = read_table(path)
        except ValueError as err:
            print(f"Пропускаю {path.name}: {err}")
            continue
        tables[key] = df
    missing = [k for k in EXPECTED if k not in tables]
    if missing:
        raise FileNotFoundError(f"В папке {folder} не хватает таблиц: {missing}")
    return tables
def validate(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """Сверяет итоги по ударам, голам и xG во всех четырёх таблицах."""
    pos = tables["position"]
    totals = {
        "Position": {
            # В Position нет числа ударов, только удары на 90 минут: восстанавливаем (с округлением)
            "Shots": (pos["Shots_p90"] * pos["Min"] / 90).sum(),
            "Goals": pos["Goals"].sum(),
            "xG": pos["xG"].sum(),
        }
    }
    for key in ("shotTypes", "shotZones", "situation"):
        df = tables[key]
        totals[CATEGORY_NAME[key]] = {
            "Shots": df["Shots"].sum(),
            "Goals": df["Goals"].sum(),
            "xG": df["xG"].sum(),
        }
    table = pd.DataFrame(totals).T.round(2)
    problems = []
    if table["Goals"].max() - table["Goals"].min() > 0.5:
        problems.append("Голы не совпадают между таблицами")
    if table["xG"].max() - table["xG"].min() > 0.1:
        problems.append("xG не совпадает между таблицами")
    exact = table.drop(index="Position")["Shots"]
    if exact.max() - exact.min() > 0.5:
        problems.append("Удары не совпадают между ShotTypes, ShotZones и Situation")
    if abs(table.loc["Position", "Shots"] - exact.mean()) > 2:
        problems.append("Удары из Position сильно расходятся с остальными таблицами")
    return table, problems
def build_summary(player: str, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pos, sit = tables["position"], tables["situation"]
    minutes = pos["Min"].sum()
    goals, assists = pos["Goals"].sum(), pos["Assists"].sum()
    xg, xa = pos["xG"].sum(), pos["xA"].sum()
    pens = sit[sit["Situation"] == "Penalty"]
    pen_goals, pen_xg, pen_shots = pens["Goals"].sum(), pens["xG"].sum(), pens["Shots"].sum()
    shots, key_passes = sit["Shots"].sum(), sit["KeyPasses"].sum()
    np_goals, np_xg, np_shots = goals - pen_goals, xg - pen_xg, shots - pen_shots
    def p90(value: float) -> float:
        return round(value / minutes * 90, 2)
    row = {
        "Player": player,
        "Apps": int(pos["Apps"].sum()),
        "Min": int(minutes),
        "Goals": int(goals),
        "PenGoals": int(pen_goals),
        "NPGoals": int(np_goals),
        "Assists": int(assists),
        "xG": round(xg, 2),
        "npxG": round(np_xg, 2),
        "xA": round(xa, 2),
        "Shots": int(shots),
        "NPShots": int(np_shots),
        "KeyPasses": int(key_passes),
        "NPGoals_p90": p90(np_goals),
        "Assists_p90": p90(assists),
        "NPG+A_p90": p90(np_goals + assists),
        "npxG_p90": p90(np_xg),
        "xA_p90": p90(xa),
        "NPShots_p90": p90(np_shots),
        "KeyPasses_p90": p90(key_passes),
        "NPGoals_minus_npxG": round(np_goals - np_xg, 2),
    }
    return pd.DataFrame([row])
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="папка с четырьмя файлами игрока (data/raw/...)")
    parser.add_argument("--player", required=True, help="имя игрока")
    parser.add_argument("--out", default="data/clean", help="корневая папка для чистых данных")
    args = parser.parse_args()
    tables = load_player(Path(args.folder))
    totals, problems = validate(tables)
    print("\nПроверка согласованности (итоги по каждой таблице):")
    print(totals.to_string())
    if problems:
        print("\nВНИМАНИЕ, найдены расхождения:")
        for p in problems:
            print(" -", p)
    else:
        print("\nИтоги совпадают, данные согласованы.")
    summary = build_summary(args.player, tables)
    print("\nСводка игрока:")
    print(summary.T.to_string(header=False))
    out_root = Path(args.out)
    player_dir = out_root / args.player.lower().replace(" ", "_")
    player_dir.mkdir(parents=True, exist_ok=True)
    for key, df in tables.items():
        df.insert(0, "Player", args.player)
        df.to_csv(player_dir / OUT_NAMES[key], index=False)
    summary.to_csv(player_dir / "summary.csv", index=False)
    # Общая таблица всех игроков, которых уже обработали
    parts = [pd.read_csv(p) for p in sorted(out_root.glob("*/summary.csv"))]
    pd.concat(parts, ignore_index=True).to_csv(out_root / "summary_all.csv", index=False)
    print(f"\nЧистые таблицы сохранены в: {player_dir}")
    print(f"Общая сводка обновлена: {out_root / 'summary_all.csv'}")
if __name__ == "__main__":
    main()
