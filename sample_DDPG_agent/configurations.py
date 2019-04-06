################################################################### 
# 
#	All the configurations are done here.
#	
#	1. Toggle the is_training flag to 0 to test a saved model.
#	2. epsilon_start is, as the name suggests, where the annealing epsilon starts from
#	3. total_explore is used as : epsilon -= 1/total_explore
#
################################################################### 

visualize_after = 5
is_training 	= 1

total_explore  	= 300000.0
max_eps 		= 8000
max_steps_eps 	= 300

wait_at_beginning 	= 0
initial_wait_period = 200		# to give the other cars a headstart of these many steps

epsilon_start  	= 1				# 
start_reward 	= -10000			# these need to be changed if restarting the playGame.py script

save_location = './'

torcsPort 	= 3101
configFile 	= '~/.torcs/config/raceman/quickrace.xml'
# configFile = '~/.torcs/config/raceman/practice.xml'
