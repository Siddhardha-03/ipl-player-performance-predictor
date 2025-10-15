<<<<<<< HEAD
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load and clean the dataset
df = pd.read_csv("ipl_data.csv")
df.replace("No stats", np.nan, inplace=True)

# Step 2: Convert numeric columns
numeric_cols = [
    'Matches_Batted', 'Not_Outs', 'Runs_Scored', 'Highest_Score', 'Batting_Average',
    'Balls_Faced', 'Batting_Strike_Rate', 'Centuries', 'Half_Centuries',
    'Fours', 'Sixes', 'Catches_Taken', 'Stumpings',
    'Matches_Bowled', 'Balls_Bowled', 'Runs_Conceded', 'Wickets_Taken',
    'Best_Bowling_Match', 'Bowling_Average', 'Economy_Rate', 'Bowling_Strike_Rate',
    'Four_Wicket_Hauls', 'Five_Wicket_Hauls'
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Step 3: Add experience feature
df["Player_Experience"] = df["Matches_Batted"].fillna(0) + df["Matches_Bowled"].fillna(0)

# Step 4: Aggregate player data
agg_df = df.groupby("Player_Name").mean(numeric_only=True).reset_index()

# Optional: Filter out players with very few matches
agg_df = agg_df[agg_df["Player_Experience"] >= 5]  # Keep experienced players only

# Step 5: Train Batting Model
batting_features = [
    'Matches_Batted', 'Not_Outs', 'Highest_Score', 'Batting_Average',
    'Balls_Faced', 'Batting_Strike_Rate', 'Centuries', 'Half_Centuries',
    'Fours', 'Sixes', 'Player_Experience'
]
batting_df = agg_df[batting_features + ['Runs_Scored']].dropna()
X_bat = batting_df[batting_features]
y_bat = batting_df['Runs_Scored']

Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_bat, y_bat, test_size=0.2, random_state=42)
bat_model = LinearRegression()
bat_model.fit(Xb_train, yb_train)

# Step 6: Train Bowling Model
bowling_features = [
    'Matches_Bowled', 'Balls_Bowled', 'Runs_Conceded', 'Bowling_Average',
    'Economy_Rate', 'Bowling_Strike_Rate', 'Four_Wicket_Hauls',
    'Five_Wicket_Hauls', 'Player_Experience'
]
bowling_df = agg_df[bowling_features + ['Wickets_Taken']].dropna()
X_bowl = bowling_df[bowling_features]
y_bowl = bowling_df['Wickets_Taken']

Xw_train, Xw_test, yw_train, yw_test = train_test_split(X_bowl, y_bowl, test_size=0.2, random_state=42)
bowl_model = LinearRegression()
bowl_model.fit(Xw_train, yw_train)

# Step 7: Repeated User Input for Prediction
print("\n🔁 Enter player names to get predictions. Type 'exit' or 'q' to quit.")

while True:
    user_input = input("\n🔍 Enter the player name (partial or full): ").strip().lower()
    if user_input in ['exit', 'q']:
        print("👋 Exiting. Thank you!")
        break

    matched_players = agg_df[agg_df["Player_Name"].str.lower().str.contains(user_input)]

    if matched_players.empty:
        print(f"\n❌ No players found matching '{user_input}'.")
    else:
        print(f"\n✅ Found {len(matched_players)} player(s) matching '{user_input}':\n")

        for _, row in matched_players.iterrows():
            player = row["Player_Name"]
            print(f"🔸 Player: {player}")

            # Batting Prediction
            bat_input = pd.DataFrame([row[batting_features].values], columns=batting_features)
            predicted_runs = bat_model.predict(bat_input)[0]
            print(f"   🏏 Predicted runs : {predicted_runs:.1f}")

            # Bowling Prediction
            bowl_input = pd.DataFrame([row[bowling_features].values], columns=bowling_features)
            predicted_wickets = bowl_model.predict(bowl_input)[0]
            print(f"   🎯 Predicted wickets: {predicted_wickets:.1f}")

            print("-" * 50)

# Step 8: Model Evaluation
y_bat_pred = bat_model.predict(Xb_test)
print("\n📊 Batting Model Evaluation:")
print(f"Mean Squared Error (MSE): {mean_squared_error(yb_test, y_bat_pred):.2f}")
print(f"R² Score: {r2_score(yb_test, y_bat_pred):.2f}")

y_bowl_pred = bowl_model.predict(Xw_test)
print("\n📊 Bowling Model Evaluation:")
print(f"Mean Squared Error (MSE): {mean_squared_error(yw_test, y_bowl_pred):.2f}")
print(f"R² Score: {r2_score(yw_test, y_bowl_pred):.2f}")
=======
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load and clean the dataset
df = pd.read_csv("ipl_data.csv")
df.replace("No stats", np.nan, inplace=True)

# Step 2: Convert numeric columns
numeric_cols = [
    'Matches_Batted', 'Not_Outs', 'Runs_Scored', 'Highest_Score', 'Batting_Average',
    'Balls_Faced', 'Batting_Strike_Rate', 'Centuries', 'Half_Centuries',
    'Fours', 'Sixes', 'Catches_Taken', 'Stumpings',
    'Matches_Bowled', 'Balls_Bowled', 'Runs_Conceded', 'Wickets_Taken',
    'Best_Bowling_Match', 'Bowling_Average', 'Economy_Rate', 'Bowling_Strike_Rate',
    'Four_Wicket_Hauls', 'Five_Wicket_Hauls'
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Step 3: Add experience feature
df["Player_Experience"] = df["Matches_Batted"].fillna(0) + df["Matches_Bowled"].fillna(0)

# Step 4: Aggregate player data
agg_df = df.groupby("Player_Name").mean(numeric_only=True).reset_index()

# Optional: Filter out players with very few matches
agg_df = agg_df[agg_df["Player_Experience"] >= 5]  # Keep experienced players only

# Step 5: Train Batting Model
batting_features = [
    'Matches_Batted', 'Not_Outs', 'Highest_Score', 'Batting_Average',
    'Balls_Faced', 'Batting_Strike_Rate', 'Centuries', 'Half_Centuries',
    'Fours', 'Sixes', 'Player_Experience'
]
batting_df = agg_df[batting_features + ['Runs_Scored']].dropna()
X_bat = batting_df[batting_features]
y_bat = batting_df['Runs_Scored']

Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_bat, y_bat, test_size=0.2, random_state=42)
bat_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
bat_model.fit(Xb_train, yb_train)

# Step 6: Train Bowling Model
bowling_features = [
    'Matches_Bowled', 'Balls_Bowled', 'Runs_Conceded', 'Bowling_Average',
    'Economy_Rate', 'Bowling_Strike_Rate', 'Four_Wicket_Hauls',
    'Five_Wicket_Hauls', 'Player_Experience'
]
bowling_df = agg_df[bowling_features + ['Wickets_Taken']].dropna()
X_bowl = bowling_df[bowling_features]
y_bowl = bowling_df['Wickets_Taken']

Xw_train, Xw_test, yw_train, yw_test = train_test_split(X_bowl, y_bowl, test_size=0.2, random_state=42)
bowl_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
bowl_model.fit(Xw_train, yw_train)

# Step 7: User Input Prediction
user_input = input("\n🔍 Enter the player name (partial or full): ").strip().lower()
matched_players = agg_df[agg_df["Player_Name"].str.lower().str.contains(user_input)]

if matched_players.empty:
    print(f"\n❌ No players found matching '{user_input}'.")
else:
    print(f"\n✅ Found {len(matched_players)} player(s) matching '{user_input}':\n")

    for _, row in matched_players.iterrows():
        player = row["Player_Name"]
        print(f"🔸 Player: {player}")

        # Batting Prediction
        bat_input = pd.DataFrame([row[batting_features].values], columns=batting_features)
        predicted_runs = bat_model.predict(bat_input)[0]
        print(f"   🏏 Predicted runs : {predicted_runs:.1f}")

        # Bowling Prediction
        bowl_input = pd.DataFrame([row[bowling_features].values], columns=bowling_features)
        predicted_wickets = bowl_model.predict(bowl_input)[0]
        print(f"   🎯 Predicted wickets: {predicted_wickets:.1f}")

        print("-" * 50)

# Step 8: Model Evaluation
y_bat_pred = bat_model.predict(Xb_test)
print("\n📊 Batting Model Evaluation:")
print(f"Mean Squared Error (MSE): {mean_squared_error(yb_test, y_bat_pred):.2f}")
print(f"R² Score: {r2_score(yb_test, y_bat_pred):.2f}")

y_bowl_pred = bowl_model.predict(Xw_test)
print("\n📊 Bowling Model Evaluation:")
print(f"Mean Squared Error (MSE): {mean_squared_error(yw_test, y_bowl_pred):.2f}")
print(f"R² Score: {r2_score(yw_test, y_bowl_pred):.2f}")
>>>>>>> b2594fced0f9dd51dfa0e1ebab16f10be6aee470
