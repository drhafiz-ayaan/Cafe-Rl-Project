import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# Dependency check
# ------------------------
try:
    import torch
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from docx import Document
except ImportError:
    print("⚠️ Missing dependencies. Install with:")
    print("pip install gymnasium stable-baselines3 torch numpy pandas matplotlib python-docx tensorboard")
    sys.exit(1)

# ------------------------
# Load and clean Rules.docx
# ------------------------
doc = Document("Rules.docx")
data = []
for table in doc.tables:
    for row in table.rows:
        data.append([cell.text.strip() for cell in row.cells])

rules_df = pd.DataFrame(data[1:], columns=data[0])
print("Rules columns detected:", list(rules_df.columns))

# Clean column names
rules_df.columns = rules_df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
rules_df = rules_df.rename(columns={
    "Service time Human (s)": "service_human_s",
    "Service time Robot (s)": "service_robot_s",
    "Automatability (1-3)": "automatability",
    "Human Interaction (0-1)": "human_interaction",
    "Priority Repetition": "priority_repetition",
    "Priority Human Interaction": "priority_human_interaction"
})

# Convert numeric cols
for c in ["service_human_s", "service_robot_s", "automatability",
          "human_interaction", "priority_repetition", "priority_human_interaction"]:
    if c in rules_df.columns:
        rules_df[c] = pd.to_numeric(rules_df[c], errors="coerce")

print("✅ Cleaned Rules:")
print(rules_df.head())

# ------------------------
# Load Orders from CSV
# ------------------------
data_dir = "data"
orders = []
for f in os.listdir(data_dir):
    if f.endswith(".csv"):
        df = pd.read_csv(os.path.join(data_dir, f))
        if "Date" in df.columns and "Time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
            orders.append(df)

if not orders:
    print("⚠️ No orders found in 'data/' folder. Exiting.")
    sys.exit(1)

orders_df = pd.concat(orders)
orders_df = orders_df.sort_values("timestamp")
print(f"✅ Loaded {len(orders_df)} orders. Time range: {orders_df['timestamp'].min()} → {orders_df['timestamp'].max()}")

# ------------------------
# Cafe Environment
# ------------------------
class CafeEnv(gym.Env):
    def __init__(self, rules, orders):
        super(CafeEnv, self).__init__()
        self.rules = rules.set_index("Drink/food")
        self.orders = orders.reset_index(drop=True)
        self.action_space = spaces.Discrete(2)  # 0=human, 1=robot
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.time = 0
        self.max_steps = len(orders)

    def reset(self, seed=None, options=None):
        self.time = 0
        obs = np.zeros(4, dtype=np.float32)
        return obs, {}

    def step(self, action):
        order = self.orders.iloc[self.time]
        drink = order.get("Drink/food", None)
        reward = 0
        done = False

        if drink in self.rules.index:
            rule = self.rules.loc[drink]
            if action == 0:  # human
                reward = -rule.get("service_human_s", 30)
            else:            # robot
                reward = -rule.get("service_robot_s", 30)

        self.time += 1
        obs = np.random.rand(4).astype(np.float32)
        if self.time >= self.max_steps:
            done = True
        return obs, reward, done, False, {}

# ------------------------
# Baselines
# ------------------------
def run_baseline(env, policy="human"):
    obs, _ = env.reset()
    total_reward = 0
    while True:
        if policy == "human":
            action = 0
        elif policy == "robot":
            action = 1
        else:  # random
            action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done: break
    return total_reward

env = CafeEnv(rules_df, orders_df)
print("Baseline human:", run_baseline(env, "human"))
print("Baseline robot:", run_baseline(env, "robot"))
print("Baseline random:", run_baseline(env, "random"))

# ------------------------
# RL Training
# ------------------------
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

monitor_file = os.path.join(log_dir, "training_monitor")
monitored_env = Monitor(env, filename=monitor_file)
vec_env = DummyVecEnv([lambda: monitored_env])

model = DQN("MlpPolicy", vec_env, verbose=1, learning_rate=1e-4, buffer_size=10000)
model.learn(total_timesteps=50000, log_interval=10)

# Save trained model
model.save("cafe_rl_model")
print("✅ Model saved as cafe_rl_model.zip")

# ------------------------
# Reward Curve Plot
# ------------------------
monitor_csv = monitor_file + ".monitor.csv"
if os.path.exists(monitor_csv):
    monitor_log = pd.read_csv(monitor_csv, skiprows=1)
    if "r" in monitor_log.columns:
        plt.figure(figsize=(10,5))
        plt.plot(monitor_log["r"])
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Cafe RL Training Reward Curve")
        plt.savefig("reward_curve.png")
        plt.close()
        print("📊 Saved reward curve to reward_curve.png")
    else:
        print("⚠️ Monitor log found but reward column missing.")
else:
    print("⚠️ No monitor log file found.")
