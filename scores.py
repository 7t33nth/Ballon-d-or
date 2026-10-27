"""
Система баллов 0-100 для сравнения игроков (Ballon d'Or критерии).

Весовые блоки:
- 50% Индивидуальная статистика
    - Результативность (NPG+A p90): 20
    - Создание моментов (xA p90 + KeyPasses p90): 15
    - Качество завершения (превышение npxG + конверсия): 10
    - Вовлечённость (удары p90): 5
- 30% Командные достижения (трофеи + индивидуальные награды)
- 20% Фейр-плей (соотношение фолов/карточек к оборонительным действиям)

Нормализация: min-max между игроками (лучший = 100, худший = 0).
"""
import pandas as pd
from pathlib import Path

# ───────────────────────── загрузка чистой статы (Understat) ─────────────────────────
summary = pd.read_csv("data/clean/summary_all.csv").set_index("Player")

# производные метрики
summary["NPG+A_p90"] = summary["NPGoals_p90"] + summary["Assists_p90"]
summary["Conversion_pct"] = (summary["NPGoals"] / summary["NPShots"] * 100).round(2)
summary["Overperformance"] = summary["NPGoals_minus_npxG"]

# ───────────────────────── загрузка FBref дисциплины ─────────────────────────
def load_fbref_discipline(path, player):
    df = pd.read_csv(path, encoding="utf-8-sig")  # CSV с запятой, разделитель по умолчанию
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    for col in ["Min", "CrdY", "CrdR", "Fls", "TklW", "Int"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["Min"].fillna(0) > 0].copy()
    minutes = df["Min"].sum()
    crdy = int(df["CrdY"].sum())
    crdr = int(df["CrdR"].sum())
    fls = int(df["Fls"].sum())
    tklw = int(df["TklW"].sum()) if "TklW" in df.columns else 0
    intr = int(df["Int"].sum()) if "Int" in df.columns else 0
    def_actions = tklw + intr
    return {
        "CrdY": crdy,
        "CrdR": crdr,
        "Fls": fls,
        "Min": int(minutes),
        "TklW": tklw,
        "Int": intr,
        "DefActions": def_actions,
        "CrdY_p90": round(crdy / minutes * 90, 3),
        "Fls_p90": round(fls / minutes * 90, 3),
        # Относительные метрики: фолы и карточки на одно оборонительное действие
        "Fls_per_DefAction": round(fls / def_actions, 3) if def_actions > 0 else 0,
        "CrdY_per_DefAction": round(crdy / def_actions, 3) if def_actions > 0 else 0,
    }

kane_disc = load_fbref_discipline(Path("data/raw/Kane/KaneFBRef.csv"), "Harry Kane")
mbappe_disc = load_fbref_discipline(Path("data/raw/Mbappe/MbappeFBRef.csv"), "Kylian Mbappe")

disc = pd.DataFrame([kane_disc, mbappe_disc], index=["Harry Kane", "Kylian Mbappe"])

# ───────────────────────── загрузка трофеев и наград ─────────────────────────
trophies = pd.read_csv("data/raw/trophies.csv")
awards = pd.read_csv("data/raw/awards.csv")

TROPHY_WEIGHTS = {
    "bundesliga": 3, "premier league": 3, "la liga": 3, "serie a": 3, "ligue 1": 3,
    "champions league": 3, "world cup": 3,
    "dfb-pokal": 2, "coppa italia": 2, "copa del rey": 2, "french cup": 2, "german cup": 2,
    "club world cup": 2, "supercopa": 1, "super cup": 1, "dfl-supercup": 1,
}

def trophy_score(player):
    df = trophies[trophies["Player"] == player]
    score = 0.0
    for _, row in df.iterrows():
        comp = str(row["Competition"]).lower()
        result = str(row["Result"]).lower()
        if result in ("winner", "champion", "won"):
            for key, w in TROPHY_WEIGHTS.items():
                if key in comp:
                    score += w
                    break
            else:
                score += 1
    return score

def award_score(player):
    df = awards[awards["Player"] == player]
    score = 0.0
    for _, row in df.iterrows():
        award = str(row["Award"]).lower()
        if "golden boot" in award or "golden shoe" in award or "butsa" in award:
            score += 3
        elif "best player" in award or "лучший игрок" in award:
            score += 2
        elif "top scorer" in award or "бомбардир" in award:
            score += 2
        elif "all-time" in award or "рекорд" in award:
            score += 1
        else:
            score += 1
    return score

# ───────────────────────── нормализация min-max ─────────────────────────
def norm(series):
    if series.max() == series.min():
        return pd.Series(50.0, index=series.index)
    return ((series - series.min()) / (series.max() - series.min()) * 100).round(1)

players = summary.index.tolist()

# индивидуальная статистика (50 блоков)
stats = pd.DataFrame(index=players)
stats["Result"] = norm(summary["NPG+A_p90"])          # 20
stats["Creation"] = norm((summary["xA_p90"] + summary["KeyPasses_p90"]) / 2)  # 15
stats["Finish"] = (norm(summary["Overperformance"]) + norm(summary["Conversion_pct"])) / 2  # 10
stats["Involvement"] = norm(summary["NPShots_p90"])   # 5

# командные достижения (30 блоков)
team = pd.DataFrame(index=players)
team["Trophies"] = [trophy_score(p) for p in players]
team["Awards"] = [award_score(p) for p in players]
team["Team_raw"] = team["Trophies"] + team["Awards"]
team["Team"] = norm(team["Team_raw"])  # 30

# фейр-плей (20 блоков)
# Чем МЕНЬШЕ фолов и карточек на одно оборонительное действие, тем ЧИЩЕ игрок.
fp = pd.DataFrame(index=players)
fp["CrdY"] = disc["CrdY"]
fp["CrdR"] = disc["CrdR"]
fp["Fls"] = disc["Fls"]
fp["DefActions"] = disc["DefActions"]
fp["Fls_per_DefAction"] = disc["Fls_per_DefAction"]
fp["CrdY_per_DefAction"] = disc["CrdY_per_DefAction"]

# Комплексный индекс грязности: карточки весим выше фолов
fp["DirtyIndex"] = fp["CrdY_per_DefAction"] * 3 + fp["Fls_per_DefAction"] * 1

# Инвертируем: чем МЕНЬШЕ грязности, тем ВЫШЕ балл
fp["FairPlay_raw"] = -fp["DirtyIndex"]
fp["FairPlay"] = norm(fp["FairPlay_raw"])  # 20

# ───────────────────────── итоговый счёт ─────────────────────────
scores = pd.DataFrame(index=players)
scores["Finishing (20)"] = (stats["Result"] / 100 * 20).round(1)
scores["Creation (15)"] = (stats["Creation"] / 100 * 15).round(1)
scores["Conversion (10)"] = (stats["Finish"] / 100 * 10).round(1)
scores["Volume (5)"] = (stats["Involvement"] / 100 * 5).round(1)
scores["Stats (50)"] = scores.iloc[:, 0:4].sum(axis=1).round(1)

scores["Team (30)"] = (team["Team"] / 100 * 30).round(1)
scores["FairPlay (20)"] = (fp["FairPlay"] / 100 * 20).round(1)

scores["TOTAL (100)"] = (scores["Stats (50)"] + scores["Team (30)"] + scores["FairPlay (20)"]).round(1)

# ───────────────────────── вывод ─────────────────────────
print("=== RAW METRICS ===")
print(summary[["Min", "Goals", "NPGoals", "xG", "npxG", "xA", "Shots", "KeyPasses",
               "NPG+A_p90", "Conversion_pct", "Overperformance"]].to_string())

print("\n=== DISCIPLINE (FBref) ===")
print(pd.concat([disc[["CrdY", "CrdR", "Fls", "TklW", "Int", "DefActions",
            "Fls_per_DefAction", "CrdY_per_DefAction"]], fp["DirtyIndex"]], axis=1).to_string())

print("\n=== TROPHIES AND AWARDS ===")
for p in players:
    print(f"{p}: trophies={trophy_score(p)}, awards={award_score(p)}")

print("\n=== NORMALIZED SCORES (0-100 within block) ===")
print(pd.DataFrame({
    "Result": stats["Result"],
    "Creation": stats["Creation"],
    "Finish": stats["Finish"],
    "Involvement": stats["Involvement"],
    "Team": team["Team"],
    "FairPlay": fp["FairPlay"],
}).to_string())

print("\n=== TOTAL SCORE 0-100 ===")
print(scores.to_string())

scores.to_csv("data/clean/scores.csv")
print("\nСохранено: data/clean/scores.csv")