r"""
Сбор данных Understat для сравнения Кейна и Мбаппе (сезон 2025/26).

Что скачивает:
1. Таблицы игроков топ-5 лиг за сезон 2025/26 (getLeagueData).
2. Пул из 10 форвардов:
   - основной: топ-10 по голам (без Кейна и Мбаппе);
   - альтернативный: топ-10 по G+A на 90 минут (мин. 1500 минут).
3. Пошотовые данные Кейна и Мбаппе (getPlayerData) — для shot maps.
4. Число ударов без пенальти для всех кандидатов пула.

Кэш: data/raw/understat/*.csv
Запуск из корня проекта:
    python fetch_understat.py
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "raw" / "understat"
SEASON = "2025"  # Understat: 2025 == сезон 2025/26

LEAGUES = ["EPL", "La liga", "Bundesliga", "Serie A", "Ligue 1"]

PLAYERS = {
    "Harry Kane": {"understat_id": "647", "file": "shots_harry_kane.csv"},
    "Kylian Mbappe": {"understat_id": "3423", "file": "shots_kylian_mbappe.csv"},
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}

NUMERIC = [
    "games", "time", "goals", "xG", "assists", "xA", "shots", "key_passes",
    "yellow_cards", "red_cards", "npg", "npxG", "xGChain", "xGBuildup",
]

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def get_json(url: str) -> dict:
    response = SESSION.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def league_url(league: str) -> str:
    return f"https://understat.com/getLeagueData/{requests.utils.quote(league)}/{SEASON}"


def player_url(player_id: str) -> str:
    return f"https://understat.com/getPlayerData/{player_id}"


def load_league_players() -> pd.DataFrame:
    frames = []
    for league in LEAGUES:
        data = get_json(league_url(league))
        frame = pd.DataFrame(data["players"])
        frame["league"] = league
        frames.append(frame)
        print(f"  {league}: {len(frame)} игроков")
        time.sleep(0.2)
    players = pd.concat(frames, ignore_index=True)
    for col in NUMERIC:
        players[col] = pd.to_numeric(players[col], errors="coerce")
    players["is_forward"] = players["position"].str.split().apply(lambda x: "F" in x)
    return players


def get_shots(player_id: str) -> pd.DataFrame:
    data = get_json(player_url(player_id))
    shots = pd.DataFrame(data["shots"])
    shots = shots[shots["season"] == SEASON].copy()
    for col in ["X", "Y", "xG", "minute"]:
        shots[col] = pd.to_numeric(shots[col], errors="coerce")
    return shots


def non_penalty_shots(player_id: str) -> int:
    shots = get_shots(player_id)
    return int((shots["situation"] != "Penalty").sum())


def select_pools(players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    forwards = players[players["is_forward"]].copy()
    forwards = forwards[
        ~forwards["player_name"].str.contains("Kane|Mbapp", case=False, na=False)
    ].copy()

    by_goals = (
        forwards.sort_values(["goals", "time"], ascending=[False, True])
        .head(10)
        .copy()
    )
    by_goals["pool_goals"] = True

    eligible = forwards[forwards["time"] >= 1500].copy()
    eligible["G+A_p90"] = (eligible["goals"] + eligible["assists"]) / eligible["time"] * 90
    by_ga = eligible.sort_values(["G+A_p90", "goals"], ascending=[False, False]).head(10).copy()
    by_ga["pool_ga90"] = True

    return by_goals, by_ga


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Скачиваю таблицы топ-5 лиг (2025/26)...")
    players = load_league_players()
    players.to_csv(OUT_DIR / "all_players_2025.csv", index=False, encoding="utf-8-sig")
    print(f"Сохранено: {OUT_DIR / 'all_players_2025.csv'} ({len(players)} строк)")

    by_goals, by_ga = select_pools(players)
    print("\nОсновной пул (топ-10 по голам):")
    print(by_goals[["player_name", "team_title", "league", "goals", "time"]].to_string(index=False))

    print("\nАльтернативный пул (топ-10 по G+A/90, мин. 1500 минут):")
    print(by_ga[["player_name", "team_title", "league", "goals", "assists", "time"]].to_string(index=False))

    candidates = pd.concat([by_goals, by_ga], ignore_index=True)
    candidates = candidates.drop_duplicates(subset=["id"]).copy()
    candidates["pool_goals"] = candidates["id"].isin(by_goals["id"])
    candidates["pool_ga90"] = candidates["id"].isin(by_ga["id"])

    print(f"\nСчитаю удары без пенальти для {len(candidates)} кандидатов пула...")
    candidates["NPShots"] = [
        non_penalty_shots(player_id) for player_id in candidates["id"]
    ]
    time.sleep(0.2)

    candidates.to_csv(OUT_DIR / "pool_candidates.csv", index=False, encoding="utf-8-sig")
    by_goals.to_csv(OUT_DIR / "pool_forwards.csv", index=False, encoding="utf-8-sig")
    by_ga.to_csv(OUT_DIR / "pool_ga90.csv", index=False, encoding="utf-8-sig")
    print(f"Сохранено: {OUT_DIR / 'pool_candidates.csv'}")
    print(f"Сохранено: {OUT_DIR / 'pool_forwards.csv'}")
    print(f"Сохранено: {OUT_DIR / 'pool_ga90.csv'}")

    print("\nСкачиваю пошотовые данные Кейна и Мбаппе...")
    for player, meta in PLAYERS.items():
        shots = get_shots(meta["understat_id"])
        shots["player_key"] = player
        shots.to_csv(OUT_DIR / meta["file"], index=False, encoding="utf-8-sig")
        print(f"  {player}: {len(shots)} ударов, {int((shots['result'] == 'Goal').sum())} голов, "
              f"xG {shots['xG'].sum():.2f} -> {meta['file']}")
        time.sleep(0.2)

    # Проверка: пошотовые суммы должны совпадать с таблицей лиги
    print("\nПроверка сумм:")
    for player, meta in PLAYERS.items():
        row = players[players["player_name"].str.contains(player.split()[-1], case=False, na=False)]
        if row.empty:
            continue
        row = row.iloc[0]
        shots = pd.read_csv(OUT_DIR / meta["file"])
        print(f"  {player}: удары {len(shots)} / {int(row['shots'])}, "
              f"голы {(shots['result'] == 'Goal').sum()} / {int(row['goals'])}, "
              f"xG {shots['xG'].sum():.2f} / {row['xG']:.2f}")

    meta = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": "2025/26",
        "sources": {
            "league_data": [league_url(league) for league in LEAGUES],
            "player_data": {
                player: player_url(meta["understat_id"])
                for player, meta in PLAYERS.items()
            },
        },
        "pool_rules": {
            "pool_goals": "топ-10 форвардов топ-5 лиг по голам, без Кейна и Мбаппе; тай-брейк — меньше минут",
            "pool_ga90": "топ-10 форвардов топ-5 лиг по (голы+ассисты)/90 при минимуме 1500 минут",
        },
        "players": PLAYERS,
    }
    (OUT_DIR / "fetch_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nСохранено: {OUT_DIR / 'fetch_meta.json'}")


if __name__ == "__main__":
    main()
