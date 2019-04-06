import sys
sys.path.append('./sample_DDPG_agent/') 

import numpy as np
np.random.seed(1337)
from gym_torcs import TorcsEnv
import snakeoil3_gym as snakeoil3
import collections as col
import random
import argparse
import tensorflow as tf
import timeit
import math
import sys
import os
from configurations import *
from ddpg import *

import gc
gc.enable()



def playGame(f_diagnostics, train_indicator, agent, port=3101):    # 1 means Train, 0 means simply Run
	
	action_dim = 3  #Steering/Acceleration/Brake
	state_dim = 65  #of sensors input
	env_name = 'Torcs_Env'
	save_location = "./weights/"

	# Generate a Torcs environment
	print("I have been asked to use port: ", port)
	env = TorcsEnv(vision=False, throttle=True, gear_change=False, main=1) 
	ob = None
	while ob is None:
		try:
			client = snakeoil3.Client(p=port, vision=False)  # Open new UDP in vtorcs
			client.MAX_STEPS = np.inf

			client.get_servers_input(0)  # Get the initial input from torcs
			obs = client.S.d  # Get the current full-observation from torcs
			ob = env.make_observation(obs)

		except:
			pass

	EXPLORE = total_explore
	episode_count = max_eps
	max_steps = max_steps_eps
	epsilon = epsilon_start
	done = False
	epsilon_steady_state = 0.01 # This is used for early stopping.
 
	totalSteps = 0
	best_reward = -100000
	running_avg_reward = 0.


	print("TORCS Experiment Start.")
	for i in range(episode_count):

		save_indicator = 0
		early_stop = 1
		total_reward = 0.
		info = {'termination_cause':0}
		distance_traversed = 0.
		speed_array=[]
		trackPos_array=[]
		
		print('\n\nStarting new episode...\n')
		print("Initial memory consumption: ")
		for step in range(max_steps):

			# Take noisy actions during training
			try:
				client.get_servers_input(step)
				snakeoil3.drive_example(client)
				client.respond_to_server()

			except Exception as e:
				print("Exception caught at port " + str(i) + str(e) )
				ob = None
				while ob is None:
					try:
						client = snakeoil3.Client(p=port, vision=False)  # Open new UDP in vtorcs
						client.MAX_STEPS = np.inf
						client.get_servers_input(0)  # Get the initial input from torcs
						obs = client.S.d  # Get the current full-observation from torcs
						ob = env.make_observation(obs)
					except:
						pass  
					continue
		
			if done:
				break

		print(info)
		try:
			if 'termination_cause' in info.keys() and info['termination_cause']=='hardReset':
				print('Hard reset by some agent')
				ob, client = env.reset(client=client, relaunch=True) 
			else:
				ob, client = env.reset(client=client, relaunch=True) 
		except Exception as e:
			print("Exception caught at point B at port " + str(i) + str(e) )
			ob = None
			while ob is None:
				try:
					client = snakeoil3.Client(p=port, vision=False)  # Open new UDP in vtorcs
					client.MAX_STEPS = np.inf
					client.get_servers_input(0)  # Get the initial input from torcs
					obs = client.S.d  # Get the current full-observation from torcs
					ob = env.make_observation(obs)
				except:
					print("Exception caught at at point C at port " + str(i) + str(e) )

	env.end()  # This is for shutting down TORCS
	print("Finish.")


def running_average(prev_avg, num_episodes, new_val):
	total = prev_avg*(num_episodes-1) 
	total += new_val
	return np.float(total/num_episodes)

def analyse_info(info, printing=True):
	simulation_state = ['Normal', 'Terminated as car is OUT OF TRACK', 'Terminated as car has SMALL PROGRESS', 'Terminated as car has TURNED BACKWARDS']
	if printing and info['termination_cause']!=0:
		print(simulation_state[info['termination_cause']])

if __name__ == "__main__":

	try:
		port = int(sys.argv[1])
	except Exception as e:
		# raise e
		print("Usage : python %s <port>" % (sys.argv[0]))
		sys.exit()

	print('is_training : ' + str(is_training))
	print('Starting best_reward : ' + str(start_reward))
	print( total_explore )
	print( max_eps )
	print( max_steps_eps )
	print( epsilon_start )
	print('config_file : ' + str(configFile))

	f_diagnostics = ""
	playGame(f_diagnostics, train_indicator=0, port=port)
