import os
import json
import pandas as pd
from glob import glob

def extract_features_from_match(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        match = json.load(f)
    
    rows = []
    info = match.get("info", {})
    venue = info.get("venue")
    match_id = os.path.splitext(os.path.basename(file_path))[0]
    toss_winner = info.get("toss", {}).get("winner")
    toss_decision = info.get("toss", {}).get("decision")
    teams = info.get("teams", [])

    # Determine target if available
    innings_data = match.get("innings", [])
    target_runs = None
    if len(innings_data) > 1:
        first_innings_runs = sum(deliv["runs"]["total"] for over in innings_data[0]["overs"] for deliv in over["deliveries"])
        target_runs = first_innings_runs + 1

    for innings_index, innings in enumerate(innings_data, start=1):
        batting_team = innings.get("team")
        overs_data = innings.get("overs", [])

        runs_so_far = 0
        wickets_so_far = 0
        balls_faced = 0

        for over in overs_data:
            over_number = over.get("over")
            for delivery in over.get("deliveries", []):
                runs_so_far += delivery["runs"]["total"]
                balls_faced += 1 if "wides" not in delivery.get("extras", {}) and "noballs" not in delivery.get("extras", {}) else 0
                if "wickets" in delivery:
                    wickets_so_far += len(delivery["wickets"])

                overs_completed = over_number + (len(over["deliveries"]) / 6.0)
                run_rate = runs_so_far / overs_completed if overs_completed > 0 else 0

                balls_remaining = None
                required_run_rate = None
                if innings_index == 2 and target_runs:
                    balls_remaining = (info.get("balls_per_over", 6) * info.get("overs", 20)) - balls_faced
                    runs_required = target_runs - runs_so_far
                    required_run_rate = (runs_required / (balls_remaining / 6)) if balls_remaining > 0 else None

                rows.append({
                    "match_id": match_id,
                    "venue": venue,
                    "toss_winner": toss_winner,
                    "toss_decision": toss_decision,
                    "batting_team": batting_team,
                    "bowling_team": [t for t in teams if t != batting_team][0] if len(teams) == 2 else None,
                    "innings": innings_index,
                    "runs_so_far": runs_so_far,
                    "wickets_so_far": wickets_so_far,
                    "balls_faced": balls_faced,
                    "run_rate": run_rate,
                    "target_runs": target_runs if innings_index == 2 else None,
                    "required_run_rate": required_run_rate,
                    "winner": info.get("outcome", {}).get("winner")  # Target label
                })
    return rows


def process_all_matches(input_folder, output_csv):
    all_rows = []
    files = glob(os.path.join(input_folder, "*.json"))
    for i, file_path in enumerate(files, start=1):
        try:
            match_rows = extract_features_from_match(file_path)
            all_rows.extend(match_rows)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        if i % 100 == 0:
            print(f"Processed {i} matches...")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")


if __name__ == "__main__":
    input_folder = r"t20s_json"
    output_csv = r"t20_data.csv"
    process_all_matches(input_folder, output_csv)
