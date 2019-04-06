## PSDDPG

This is a sample implementation of PS-DDPG ( Parameter Sharing Deep Deterministic Policy Gradient)
for Reinforcement learning . It has been used for simulating lanekeeping behaviour of cars/agents in [Gym-TORCS](https://github.com/ugo-nama-kun/gym_torcs). 

#### Requirements: 
- vtorcs(https://github.com/giuse/vtorcs/tree/nosegfault)
- Python 3
- Tensorflow
- xautomation
- openai-gym

#### How-To:

After setting up vtorcs, replace ```~/vtorcs``` and ```~/autostart.sh``` with their absolute locations.
Make sure the number of agents/scr_servers in torcs is same as mentioned in multi_ddpg.py. 

For running the code:
```
python3 multi_ddpg.py
```

In a new terminal, 
```
sh startTorcs.sh 
```
For training, set  ```train_indicator=1```, in multi_ddpg.py  and ```train_indicator=0``` for testing.










