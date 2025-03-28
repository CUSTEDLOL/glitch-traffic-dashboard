# 🚦 GLITCH - Guided Live Intelligent Traffic Control Hub

GLITCH is an AI-powered, real-time traffic management dashboard designed to optimize urban signal control using reinforcement learning and live data. It simulates and compares smart (AI-based) and traditional signal behavior, offers visual traffic analytics, and integrates live camera feeds from Singapore.

---

## 🌟 Features

- 🛰️ **Live Traffic Visualization** with simulated congestion dots and LTA camera feeds
- 🧠 **Reinforcement Learning Agent** optimizing signal choices at key junctions
- 📊 **Before vs After Analytics** comparing traditional vs AI-managed intersections
- 📍 **RL-Controlled Map** showing AI behavior across multiple regions
- 📈 **Trend Charts** for reward performance over time
- 📋 **Contingency Testing** for statistical pattern differences
- 💬 **AI Traffic Assistant** powered by Gemini for interactive queries

---

## 📍 Covered Junctions
- Orchard Rd
- Yishun Ave 2
- Tampines St 81
- Clementi Rd
- Jurong West Ave 5
- Upper Thomson Rd
- Bedok North Ave 3
- Bukit Timah Rd
- Choa Chu Kang Ave 1
- Punggol Field

---

## 📦 Tech Stack

- **Frontend**: Streamlit + Folium
- **Backend**: Python (Reinforcement Learning, Pandas, Scipy)
- **APIs**: Google Gemini (AI assistant), Singapore LTA Traffic Images
- **Libraries**: `streamlit-folium`, `scipy`, `numpy`, `requests`

---

## 🚀 Deployment (Streamlit Cloud)

1. Clone the repo
2. Add `simulation_log.csv` in the root folder
3. Create `requirements.txt` (already provided)
4. Deploy on [Streamlit Cloud](https://streamlit.io/cloud)

---

## 🔐 Gemini API Key Setup

Create a `.streamlit/secrets.toml` file:

```toml
[gemini]
api_key = "YOUR_GEMINI_API_KEY"
