import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load and clean the dataset
df = pd.read_csv("ipl_data.csv")
df.replace("No stats", np.nan, inplace=True)

# Convert relevant columns to numeric and fill empty cells with 0
cols = ['Matches_Batted', 'Batting_Average', 'Runs_Scored',
        'Matches_Bowled', 'Bowling_Average', 'Wickets_Taken']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# Step 2: Aggregate player data across seasons
agg_df = df.groupby("Player_Name")[cols].mean().reset_index()

# Thresholds
threshold_bat_matches = 5
threshold_bowl_matches = 5
threshold_wickets = 10

# Step 3: Batting Model (players with enough batting matches)
batting_df = agg_df[agg_df['Matches_Batted'] >= threshold_bat_matches]
batting_features = ['Matches_Batted', 'Batting_Average']
X_bat = batting_df[batting_features]
y_bat = batting_df['Runs_Scored']

Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_bat, y_bat, test_size=0.2, random_state=42)
bat_model = LinearRegression()
bat_model.fit(Xb_train, yb_train)

# Step 4: Bowling Model (players with enough bowling matches and wickets)
bowling_df = agg_df[(agg_df['Matches_Bowled'] >= threshold_bowl_matches) & (agg_df['Wickets_Taken'] >= threshold_wickets)]
bowling_features = ['Matches_Bowled', 'Bowling_Average']
X_bowl = bowling_df[bowling_features]
y_bowl = bowling_df['Wickets_Taken']

Xw_train, Xw_test, yw_train, yw_test = train_test_split(X_bowl, y_bowl, test_size=0.2, random_state=42)
bowl_model = LinearRegression()
bowl_model.fit(Xw_train, yw_train)

# Step 5: User Input for Prediction
print("\n🔁 Enter a player name to get prediction. Type 'exit' to quit.")

while True:
    user_input = input("\n🔍 Enter player name: ").strip().lower()
    if user_input in ['exit', 'q']:
        print("👋 Exiting.")
        break

    matched = agg_df[agg_df["Player_Name"].str.lower().str.contains(user_input)]

    if matched.empty:
        print(f"❌ No player found matching '{user_input}'")
        continue

    for _, row in matched.iterrows():
        name = row['Player_Name']
        matches_batted = row['Matches_Batted']
        matches_bowled = row['Matches_Bowled']
        wickets_taken = row['Wickets_Taken']

        print(f"\n🔸 Player: {name}")

        is_batsman = matches_batted >= threshold_bat_matches
        is_bowler = (matches_bowled >= threshold_bowl_matches) and (wickets_taken >= threshold_wickets)

        if is_batsman and not is_bowler:
            # Batsman only
            bat_input = pd.DataFrame([[matches_batted, row['Batting_Average']]], columns=batting_features)
            pred_runs = bat_model.predict(bat_input)[0]
            print(f"   🏏 Predicted Runs (avg season): {pred_runs:.1f}")

        elif is_bowler and not is_batsman:
            # Bowler only
            bowl_input = pd.DataFrame([[matches_bowled, row['Bowling_Average']]], columns=bowling_features)
            pred_wickets = bowl_model.predict(bowl_input)[0]
            print(f"   🎯 Predicted Wickets (avg season): {pred_wickets:.1f}")

        elif is_batsman and is_bowler:
            # Allrounder: predict both
            bat_input = pd.DataFrame([[matches_batted, row['Batting_Average']]], columns=batting_features)
            pred_runs = bat_model.predict(bat_input)[0]
            bowl_input = pd.DataFrame([[matches_bowled, row['Bowling_Average']]], columns=bowling_features)
            pred_wickets = bowl_model.predict(bowl_input)[0]

            print(f"   ⚖️ Allrounder detected.")
            print(f"   🏏 Predicted Runs (avg season): {pred_runs:.1f}")
            print(f"   🎯 Predicted Wickets (avg season): {pred_wickets:.1f}")

        else:
            print("   ℹ️ Not enough data to classify as batsman or bowler.")

        print("-" * 40)

# Step 6: Show Model Evaluation at the end
print("\n📊 Batting Model Evaluation:")
print(f"   Mean Squared Error (MSE): {mean_squared_error(yb_test, bat_model.predict(Xb_test)):.2f}")
print(f"   R² Score: {r2_score(yb_test, bat_model.predict(Xb_test)):.2f}")

print("\n📊 Bowling Model Evaluation:")
print(f"   Mean Squared Error (MSE): {mean_squared_error(yw_test, bowl_model.predict(Xw_test)):.2f}")
print(f"   R² Score: {r2_score(yw_test, bowl_model.predict(Xw_test)):.2f}")
