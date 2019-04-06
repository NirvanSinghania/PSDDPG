import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("3101.csv")
x = df['ep_no']
y = df['avg_reward_without_back']
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.plot(x,y)
plt.show()
