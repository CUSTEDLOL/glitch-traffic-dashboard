\
# final_dashboard_complete.py
import requests
import streamlit as st
import pandas as pd
import folium
import random
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from collections import Counter
from scipy.stats import chi2_contingency
import google.generativeai as genai
import numpy as np



st.set_page_config(layout="wide")
st.title("🚦 GLITCH - Guided Live Intelligent Traffic Control Hub")
st.write("A Smart Traffic AI Dashboard")

# === JUNCTIONS ===
junctions = [
    ("Orchard Rd", 1.3048, 103.8318),
    ("Yishun Ave 2", 1.4295, 103.8355),
    ("Tampines St 81", 1.3496, 103.9366),
    ("Clementi Rd", 1.3150, 103.7700),
    ("Jurong West Ave 5", 1.3403, 103.7064),
    ("Upper Thomson Rd", 1.3538, 103.8326),
    ("Bedok North Ave 3", 1.3286, 103.9323),
    ("Bukit Timah Rd", 1.3243, 103.8177),
    ("Choa Chu Kang Ave 1", 1.3849, 103.7455),
    ("Punggol Field", 1.3937, 103.9134)
]

# === RL ENV ===
class SimpleTrafficEnv:
    def __init__(self, max_cars=10):
        self.max_cars = max_cars
        self.num_lanes = 4
        self.state = [random.randint(0, max_cars) for _ in range(self.num_lanes)]
        self.action_space = [0, 1, 2, 3]
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.eps = 0.2
        self.logs = []

    def get_state_key(self, state):
        return tuple(state)

    def choose_action(self, state):
        key = self.get_state_key(state)
        if random.random() < self.eps or key not in self.q_table:
            return random.choice(self.action_space)
        return max(self.q_table[key], key=self.q_table[key].get)

    def step(self, action):
        queues = self.state.copy()
        queues[action] = max(0, queues[action] - random.randint(2, 4))
        for i in range(4):
            if i != action:
                queues[i] = min(self.max_cars, queues[i] + random.randint(0, 2))
        reward = -sum(queues)
        return queues, reward

    def update_q(self, state, action, reward, next_state):
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        if key not in self.q_table:
            self.q_table[key] = {a: 0 for a in self.action_space}
        if next_key not in self.q_table:
            self.q_table[next_key] = {a: 0 for a in self.action_space}
        self.q_table[key][action] += self.alpha * (
            reward + self.gamma * max(self.q_table[next_key].values()) - self.q_table[key][action])

    def simulate(self, steps=100, mode="AI"):
        self.logs = []
        state = self.state.copy()
        for _ in range(steps):
            if mode == "AI":
                action = self.choose_action(state)
            else:
                action = random.choice(self.action_space)
            next_state, reward = self.step(action)
            if mode == "AI":
                self.update_q(state, action, reward, next_state)
            self.logs.append({
                "queues": state.copy(),
                "green": ["North", "South", "East", "West"][action],
                "reward": reward
            })
            state = next_state
        return self.logs

# === SIMULATE DATA ===
if "simulations" not in st.session_state:
    st.session_state["simulations"] = {}
    for name, lat, lon in junctions:
        env_ai = SimpleTrafficEnv()
        env_dumb = SimpleTrafficEnv()
        ai_logs = env_ai.simulate(mode="AI")
        dumb_logs = env_dumb.simulate(mode="DUMB")
        st.session_state["simulations"][name] = {
            "coords": (lat, lon),
            "AI": ai_logs,
            "DUMB": dumb_logs
        }

# === TABS ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🛰️ Live Map Data", "📍 RL Training Map", "📊 Before vs After", "⏪ Replay", "📈 Trend", "📋 Contingency Test", "🧠 Advisor"
])

# === LIVE MAP TAB ===


# === LIVE MAP TAB ===
with tab1:
    st.subheader("🛰️ Real-Time Traffic Dots + Camera Feed")

    import requests  # ensure requests is defined

    @st.cache_data(ttl=60)
    def fetch_camera_data():
        try:
            res = requests.get("https://api.data.gov.sg/v1/transport/traffic-images")
            return res.json()["items"][0]["cameras"]
        except Exception as e:
            st.error(f"Error fetching camera data: {e}")
            return []

    camera_data = fetch_camera_data()

    fake_points = [
        (1.2951, 103.8580), (1.3521, 103.8198), (1.3496, 103.9568), (1.2902, 103.8372),
        (1.3781, 103.8494), (1.2842, 103.8430), (1.3345, 103.7467), (1.3125, 103.8869),
        (1.3795, 103.7607), (1.3198, 103.6976), (1.3104, 103.7915), (1.3280, 103.8532)
    ]

    if "realtime_dots" not in st.session_state:
        st.session_state["realtime_dots"] = [
            {"lat": lat, "lon": lon, "color": random.choice(["red", "orange", "green"])}
            for lat, lon in fake_points
        ]

    m_live = folium.Map(location=[1.3521, 103.8198], zoom_start=12)

    # Simulated traffic dots
    for dot in st.session_state["realtime_dots"]:
        folium.CircleMarker(
            location=(dot["lat"], dot["lon"]),
            radius=7,
            color=dot["color"],
            fill=True,
            fill_opacity=0.6,
            popup=f"Simulated Traffic: {dot['color'].upper()}"
        ).add_to(m_live)

    # Live camera feeds
    df_cam = pd.DataFrame(camera_data)
    df_cam["latitude"] = df_cam["location"].apply(lambda x: x["latitude"])
    df_cam["longitude"] = df_cam["location"].apply(lambda x: x["longitude"])

    for _, row in df_cam.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        cam_popup = folium.Popup(
            f"<b>Camera ID:</b> {row['camera_id']}<br><b>Time:</b> {row['timestamp']}<br><img src='{row['image']}' width='200px'>",
            max_width=250
        )
        folium.Marker(
            location=(lat, lon),
            icon=folium.Icon(color='blue', icon='camera', prefix='fa'),
            popup=cam_popup
        ).add_to(m_live)

    st_folium(m_live, width=1000, height=650)


# === MAP TAB ===
with tab2:
    m = folium.Map(location=[1.3521, 103.8198], zoom_start=12)
    color_map = {"North": "green", "South": "blue", "East": "orange", "West": "red"}
    for name, data in st.session_state["simulations"].items():
        lat, lon = data["coords"]
        latest = data["AI"][-1]
        popup = f"{name}"
        color = color_map.get(latest["green"], "gray")
        folium.Marker(
            location=(lat, lon),
            icon=folium.Icon(color=color, icon="cog", prefix="fa"),
            popup=popup,
            tooltip=f"{name}: Click to view RL log"
        ).add_to(m)
    st.subheader("🗺️ RL-Controlled Junctions")
    result = st_folium(m, width=1000, height=600)
    selected = result.get("last_object_clicked_popup")
    if selected:
        st.subheader(f"📄 RL Log for {selected}")
        st.dataframe(st.session_state["simulations"][selected]["AI"])

# === BEFORE VS AFTER ===
with tab3:
    st.subheader("📊 Before vs After AI")
    summary = []
    for name, sim in st.session_state["simulations"].items():
        q_dumb = sum(-x["reward"] for x in sim["DUMB"])
        q_ai = sum(-x["reward"] for x in sim["AI"])
        summary.append((name, q_dumb, q_ai, q_dumb - q_ai))
    df = pd.DataFrame(summary, columns=["Junction", "Dumb Total Queue", "AI Total Queue", "Improvement"])
    st.dataframe(df)
    st.bar_chart(df.set_index("Junction")[["Dumb Total Queue", "AI Total Queue"]])

# === REPLAY ===
with tab4:
    st.subheader("⏪ RL Replay")
    junc = st.selectbox("Choose junction", [j[0] for j in junctions])
    step = st.slider("Step", 0, 99, 0)
    data = st.session_state["simulations"][junc]["AI"][step]
    st.write(f"Step {step}: Green = {data['green']}")
    st.write(f"Queues: {data['queues']}")
    st.write(f"Reward: {data['reward']}")

# === TRENDS ===
with tab5:
    st.subheader("📈 Reward Trend")

    junc_trend = st.selectbox("Pick junction for trend", [j[0] for j in junctions], key="trend_selector")
    ai = [x["reward"] for x in st.session_state["simulations"][junc_trend]["AI"]]
    dumb = [x["reward"] for x in st.session_state["simulations"][junc_trend]["DUMB"]]
    st.line_chart(pd.DataFrame({"AI": ai, "Dumb": dumb}))



# === ADVISOR ===
with tab7:
    st.subheader("🧠 AI Traffic Planner")
    advisor_junction = st.selectbox("Select a junction", [j[0] for j in junctions], key="advisor")
    queues = st.session_state["simulations"][advisor_junction]["AI"][-1]["queues"]
    total_queue = sum(queues)
    max_lane = ["North", "South", "East", "West"][queues.index(max(queues))]
    st.write(f"Queue: {queues}")
    st.write(f"Total: {total_queue}, Worst: {max_lane}")
    st.markdown("### Recommendations:")
    suggestions = []
    if max(queues) >= 9:
        suggestions.append(f"🔴 Add a **bus stop** or drop-off bay near **{max_lane}**.")
    if total_queue >= 35:
        suggestions.append("🚧 Consider a **flyover** or U-turn loop.")
    if queues.count(10) >= 2:
        suggestions.append("🚴 Add **cycle lanes** nearby.")
    if all(q >= 6 for q in queues):
        suggestions.append("🛣️ Optimize signal timing or build a feeder road.")
    if not suggestions:
        suggestions.append("✅ Load manageable. No urgent action.")
    for s in suggestions:
        st.markdown(s)


with tab6:
    st.subheader("📊 2-Way Paired Contingency Test")
    st.write("Compare green light distribution between two junctions.")

    junction_names = list(st.session_state["simulations"].keys())

    j1 = st.selectbox("Select Junction 1", junction_names, key="j1")
    j2 = st.selectbox("Select Junction 2", junction_names, key="j2")

    if st.button("Run Test"):
        data1 = [log["green"] for log in st.session_state["simulations"][j1]["AI"]]
        data2 = [log["green"] for log in st.session_state["simulations"][j2]["AI"]]

        lanes = ["North", "South", "East", "West"]
        count1 = Counter(data1)
        count2 = Counter(data2)

        obs_matrix = [[count1.get(lane, 0), count2.get(lane, 0)] for lane in lanes]
        df_obs = pd.DataFrame(obs_matrix, index=lanes, columns=[j1, j2])

        st.write("### Observed Green Light Frequencies")
        st.dataframe(df_obs)

        chi2, p, dof, expected = chi2_contingency(df_obs)

        st.markdown(f"**Chi-squared Statistic**: {chi2:.2f}")
        st.markdown(f"**Degrees of Freedom**: {dof}")
        st.markdown(f"**p-value**: {p:.4f}")

        if p < 0.05:
            st.success("✅ Statistically significant difference found between the junctions' light patterns.")
        else:
            st.info("ℹ️ No significant difference found between the junctions.")



#Chatbot

# === GEMINI CONFIG ===

genai.configure(api_key=st.secrets["gemini"]["api_key"])
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")  

# === LOAD SIMULATION LOG ===
@st.cache_data
def load_simulation_log():
    path = r"simulation_log.csv"
    df = pd.read_csv(path)
    return df

log_df = load_simulation_log()
lanes = ["North", "South", "East", "West"]

# === INSIGHT FUNCTIONS ===
def describe_step(n):
    if 0 <= n < len(log_df):
        row = log_df.iloc[n]
        green = lanes[int(row["green_lane"])]
        queues = eval(row["queues"])
        return f"Step {n}: Queues = {queues}, Green light = {green}."
    return f"Step {n} not found."

def count_green_lanes():
    counts = log_df["green_lane"].value_counts().to_dict()
    return {lanes[k]: v for k, v in counts.items()}

def find_max_congestion():
    return [idx for idx, row in log_df.iterrows() if all(val == 10 for val in eval(row["queues"]))]

def average_queue():
    avg = np.mean(log_df["queues"].apply(eval).tolist(), axis=0)
    return dict(zip(lanes, avg))

def worst_lane():
    avg = average_queue()
    lane = max(avg, key=avg.get)
    return f"The worst lane is {lane} (avg queue: {avg[lane]:.2f})."

def smartest_decision():
    log_df["sum"] = log_df["queues"].apply(lambda x: sum(eval(x)))
    min_row = log_df.loc[log_df["sum"].idxmin()]
    return f"Step {int(min_row['step'])} had the lowest total queue ({int(min_row['sum'])})."

def rl_pattern_summary():
    series = log_df["green_lane"].astype(int)
    switches = (series.diff().fillna(0) != 0).sum()
    frequent = series.mode()[0]
    return f"RL switched lanes {switches} times. Most frequent: {lanes[frequent]}."

# === CHATBOT ===
st.subheader("💬 Ask the AI Traffic Assistant")
user_input = st.chat_input("Ask about traffic, steps, or congestion...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    if "step" in user_input:
        step = int("".join(filter(str.isdigit, user_input)))
        insight = describe_step(step)
    elif "green light" in user_input:
        insight = ", ".join([f"{k}: {v}" for k, v in count_green_lanes().items()])
    elif "maxed out" in user_input or "all queues 10" in user_input:
        steps = find_max_congestion()
        insight = f"Steps with all queues maxed: {steps}" if steps else "No maxed steps."
    elif "average" in user_input:
        avg = average_queue()
        insight = ", ".join([f"{k}: {v:.2f}" for k, v in avg.items()])
    elif "worst lane" in user_input:
        insight = worst_lane()
    elif "smartest" in user_input:
        insight = smartest_decision()
    elif "pattern" in user_input or "RL" in user_input:
        insight = rl_pattern_summary()
    else:
        insight = "Try asking about: step 5, average queue, green light count, maxed queues, RL pattern."

    prompt = f"Use this insight to help the user: {insight}\n\nUser: {user_input}"

    with st.chat_message("assistant"):
        answer = model.generate_content(prompt)
        st.markdown(answer.text)
        
        
        
        