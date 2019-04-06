import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import playGame_DDPG
import helpers
import os
from random import choice
from time import sleep
from time import time
import snakeoil3_gym as snakeoil3
import sys
sys.path.append('./sample_DDPG_agent/')
from ddpg import *

with tf.device("/cpu:0"): 
        #Configure the number of workers and helpers required
        num_workers = 3 
        num_nonworkers = 0
        print("numb of workers is" + str(num_workers))
      
        worker_threads = []
        action_dim = 3  #Steering/Acceleration/Brake
        state_dim = 65  #of sensors input
        env_name = 'Torcs_Env'
        save_location = "./weights/" 
        agent = DDPG(env_name, state_dim, action_dim, save_location)
  
        for i in range(num_nonworkers):
                worker_work = lambda: (helpers.playGame(f_diagnostics=""+str(i), train_indicator=0, \
                 agent=None,port=3101+i))
                print("hi i am here \n")
                t = threading.Thread(target=(worker_work))
                print("active thread count is: " + str(threading.active_count()) + "\n")
                t.start()
                sleep(0.5)
                worker_threads.append(t)
                
        for i in range(num_workers):
                worker_work = lambda:(playGame_DDPG.playGame(f_diagnostics=""+str(i), train_indicator=1, \
                 agent=agent, port=3101+i+num_nonworkers))
                print("hi i am here \n")
                t = threading.Thread(target=(worker_work))
                print("active thread count is: " + str(threading.active_count()) + "\n")
                t.start()
                sleep(0.5)
                worker_threads.append(t)

        
