import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv("ipl_data.csv")

# Step 2: Clean the data
df.replace("No stats", np.nan, inplace=True)

# Convert numeric columns
numeric_cols = [
    'Matches_Batted', 'Not_Outs', 'Runs_Scored', 'Highest_Score', 'Batting_Average',
    'Balls_Faced', 'Batting_Strike_Rate', 'Centuries', 'Half_Centuries',
    'Fours', 'Sixes', 'Catches_Taken', 'Stumpings',
    'Matches_Bowled', 'Balls_Bowled', 'Runs_Conceded', 'Wickets_Taken',
    'Best_Bowling_Match', 'Bowling_Average', 'Economy_Rate', 'Bowling_Strike_Rate',
    'Four_Wicket_Hauls', 'Five_Wicket_Hauls'
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Step 3: Train Batting Model (Predict Runs_Scored)
batting_features = [
    'Matches_Batted', 'Not_Outs', 'Highest_Score', 'Batting_Average',
    'Balls_Faced', 'Batting_Strike_Rate', 'Centuries', 'Half_Centuries',
    'Fours', 'Sixes'
]
batting_df = df[batting_features + ['Runs_Scored']].dropna()

X_bat = batting_df[batting_features]
y_bat = batting_df['Runs_Scored']

Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_bat, y_bat, test_size=0.2, random_state=42)

bat_model = RandomForestRegressor(n_estimators=100, random_state=42)
bat_model.fit(Xb_train, yb_train)

# Step 4: Train Bowling Model (Predict Wickets_Taken)
bowling_features = [
    'Matches_Bowled', 'Balls_Bowled', 'Runs_Conceded', 'Bowling_Average',
    'Economy_Rate', 'Bowling_Strike_Rate', 'Four_Wicket_Hauls', 'Five_Wicket_Hauls'
]
bowling_df = df[bowling_features + ['Wickets_Taken']].dropna()

X_bowl = bowling_df[bowling_features]
y_bowl = bowling_df['Wickets_Taken']

Xw_train, Xw_test, yw_train, yw_test = train_test_split(X_bowl, y_bowl, test_size=0.2, random_state=42)

bowl_model = RandomForestRegressor(n_estimators=100, random_state=42)
bowl_model.fit(Xw_train, yw_train)

# Step 5: Ask user for partial or full player name
user_input = input("\n🔍 Enter the player name (partial or full): ").strip().lower()

# Match all players whose names contain the input (case-insensitive)
matched_players = df[df["Player_Name"].str.lower().str.contains(user_input)]

if matched_players.empty:
    print(f"\n❌ No players found matching '{user_input}'.")
else:
    unique_players = matched_players["Player_Name"].unique()
    print(f"\n✅ Found {len(unique_players)} player(s) matching '{user_input}':\n")

    for player in unique_players:
        print(f"🔸 Player: {player}")
        player_data = df[df["Player_Name"] == player]

        # --- Batting Prediction ---
        bat_input = player_data[batting_features].dropna()
        if not bat_input.empty:
            predicted_runs = bat_model.predict(bat_input)[0]
            print(f"   🏏 Predicted Runs Scored: {predicted_runs:.1f}")
        else:
            print("   ⚠️ Insufficient batting data.")

        # --- Bowling Prediction ---
        bowl_input = player_data[bowling_features].dropna()
        if not bowl_input.empty:
            predicted_wickets = bowl_model.predict(bowl_input)[0]
            print(f"   🎯 Predicted Wickets Taken: {predicted_wickets:.1f}")
        else:
            print("   ⚠️ Insufficient bowling data.")

        print("-" * 50)




#additional features

        
# Batting model evaluation
y_bat_pred = bat_model.predict(Xb_test)
print("🏏 Batting Model:")
print("MSE:", mean_squared_error(yb_test, y_bat_pred))
print("R² Score:", r2_score(yb_test, y_bat_pred))

# Bowling model evaluation
y_bowl_pred = bowl_model.predict(Xw_test)
print("\n🎯 Bowling Model:")
print("MSE:", mean_squared_error(yw_test, y_bowl_pred))
print("R² Score:", r2_score(yw_test, y_bowl_pred))








        
