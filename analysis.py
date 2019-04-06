import pandas as pd

num_agents = 6
A = []
for i in range(num_agents):
	A.append(pd.read_csv(str(3101+i)+".csv"))

# num_episodes = int(len(A[0]))*(int(A[0]["tot_steps"][0]))
num_episodes = int(len(A[0]))


avg_pure_reward = 0
avg_reward = 0
avg_progress = 0

for i in range(num_agents):
	df = A[i]
	avg_pure_reward += df["tot_pure_reward"].sum()/num_episodes
	avg_reward += df["tot_reward"].sum()/num_episodes
	avg_progress += df["tot_progress"].sum()/num_episodes
print(avg_pure_reward/num_agents, avg_reward/num_agents, avg_progress/num_agents,sep=",")


avg_pure_reward_vel = 0 
avg_reward_vel = 0 
avg_progress_vel = 0
for i in range(num_agents):
	df = A[i]
	avg_pure_reward_vel += df["avg_pure_reward_vel"].sum()/num_episodes
	avg_reward_vel += df["avg_reward_vel"].sum()/num_episodes
	avg_progress_vel += df["avg_progress_vel"].sum()/num_episodes

print(avg_pure_reward_vel/num_agents, avg_reward_vel/num_agents, avg_progress_vel/num_agents,sep=",")

avg_collisions = 0 
avg_racepos = 0
for i in range(num_agents):
	df = A[i]
	avg_collisions += df["avg_collisions"].sum()/num_episodes
	avg_racepos += df["self.racepos"].sum()/num_episodes

print(avg_collisions/num_agents,avg_racepos/num_agents,sep=",")



