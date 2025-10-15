# 🏏 IPL Player Performance Predictor

Predict **batting and bowling performance** of IPL players using machine learning.  
Search by **full or partial player names** (like `Virat`, `Sharma`, etc.) and get estimated stats like **runs scored** and **wickets taken**.

---

## 🚀 Features

- 🔍 **Partial name search**: Find all matching players with a keyword like `"sharma"` or `"virat"`.
- 🧠 **ML-powered predictions**: Uses Random Forest Regression to predict:
  - 🏏 Runs Scored (batting)
  - 🎯 Wickets Taken (bowling)
- ✅ Handles missing data gracefully
- 📊 Shows personalized predictions for each player found
- 💡 Easy to extend into a web app using Streamlit or Flask

🔍 Enter the player name (partial or full): sharma

✅ Found 4 player(s) matching 'sharma':

🔸 Player: Rohit Sharma
   🏏 Predicted Runs Scored: 482.7
   ⚠️ Insufficient bowling data.
--------------------------------------------------
🔸 Player: Sandeep Sharma
   ⚠️ Insufficient batting data.
   🎯 Predicted Wickets Taken: 17.2
--------------------------------------------------
🔸 Player: Karan Sharma
   🏏 Predicted Runs Scored: 120.4
   🎯 Predicted Wickets Taken: 7.1
--------------------------------------------------
🔸 Player: Ishant Sharma
   ⚠️ Insufficient batting data.
   🎯 Predicted Wickets Taken: 12.6
--------------------------------------------------

