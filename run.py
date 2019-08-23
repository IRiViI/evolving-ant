import roboschool
import gym
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import argparse

from tensorflow.keras.layers import Dense, Input, LeakyReLU, concatenate, Lambda, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K

from antAgent import Ant

if __name__ == '__main__':


	parser = argparse.ArgumentParser()
	parser.add_argument('--steps')
	parser.add_argument('--creatures')
	parser.add_argument('--hidden_layers')
	parser.add_argument('--models_dir')
	args = parser.parse_args()

	my_dict = {'steps': args.steps, 'creatures': args.creatures, 'hidden_layers': args.hidden_layers}
	print(my_dict)

	# Settings
	max_number_of_steps = 600 if args.steps == None else int(args.steps)
	number_of_creatures = 300 if args.creatures == None else int(args.creatures)
	hidden_layer_widths = [64,32] if args.hidden_layers == None else [int(i) for i in args.hidden_layers.split(',')]
	model_directory = './agents_weights/64_32' if args.models_dir == None else args.models_dir

	agent_index = random.randint(0,number_of_creatures-1)

	env = gym.make('RoboschoolAnt-v1')
	env.reset()
	action = env.action_space.sample()
	obv, reward, is_done, info = env.step(action)

	input_shape = np.array(obv).shape
	output_shape = np.array(action).shape
	    
	agent = Ant(input_shape, output_shape, 
	                hidden_layer_widths = hidden_layer_widths)
	agent.build()

	agent.model.load_weights('{}/{}.h5'.format(model_directory, agent_index))        

	total_reward = 0 

	obv = env.reset()
	obvs = [obv]
	obvs = np.array(obvs)

	for step_number in range(max_number_of_steps):
	    action = agent.predict(obvs)[0]

	    obv, reward, is_done, info = env.step(action)
	    
	    obvs = [obv]
	    obvs = np.array(obvs)

	    total_reward += reward

	    env.render()

	    time.sleep(0.01)