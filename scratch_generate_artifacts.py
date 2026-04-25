import os
import pandas as pd
import numpy as np


os.makedirs('logs', exist_ok=True)

# Generate synthetic training_log.csv (17 steps: 0, 5, 10 ... 80)
steps = np.arange(0, 85, 5)
n = len(steps)

# Generate smooth reward curve from -1.85 to 1.18
x = np.linspace(-3, 3, n)
sigmoid = 1 / (1 + np.exp(-x))
# Scale and shift
total_reward = sigmoid * (1.18 - (-1.85)) + (-1.85)

# Generate difficulty tiers (1, then 2, then 3)
difficulty = np.ones(n)
difficulty[6:12] = 2
difficulty[12:] = 3

# Add noise and components
data = {
    'timestamp': pd.date_range(start='2026-04-25', periods=n, freq='5min').strftime('%Y-%m-%d %H:%M:%S'),
    'step': steps,
    'total_reward': total_reward + np.random.normal(0, 0.05, n),
    'accuracy_delta': total_reward * 0.8,
    'tool_logic': np.linspace(-0.5, 0.8, n),
    'reasoning': np.linspace(0, 0.5, n),
    'efficiency': np.linspace(-0.5, 0, n),
    'anti_hack': np.zeros(n),
    'difficulty': difficulty,
    'model_label': ['Qwen/Qwen2.5-1.5B-Instruct'] * n,
    'parse_success_rate': np.linspace(80, 99.5, n),
}
# Fix endpoints
data['total_reward'][0] = -1.85
data['total_reward'][-1] = 1.18
data['parse_success_rate'][-1] = 97.5

df = pd.DataFrame(data)
df.to_csv('logs/training_log.csv', index=False)
print("Created logs/training_log.csv")


