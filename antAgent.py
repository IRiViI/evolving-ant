import roboschool
import gym
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import time
tf.compat.v1.disable_eager_execution()

from tensorflow.keras.layers import Dense, Input, LeakyReLU, concatenate, Lambda, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K

class Ant():
    
    def __init__(self, input_shape, output_shape, 
    	hidden_layer_widths = [64,32]):
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        self.output_layer_activation = 'tanh'
        self.hidden_layer_activation = 'relu'
        
        self.input = None
        self.model = None
        
        self.states = []
        self.actions = []
        self.rewards = []
        
        self.mutation_chance = 0.02
        self.mutation_distance = 0.005

        self.hidden_layer_widths = hidden_layer_widths
        
        self.data = DataObject({
            'parents':[],
            'fitness_value': 0,
            'mutated': False,
            'cloned': False
        })
        
    def build(self):
        
        # Create input layer
        self.input = Input(shape=self.input_shape)
        input = self.input

        # Create hidden layers
        for hiddenlayer_index, hiddenlayer_width in enumerate(self.hidden_layer_widths):
        	layer = Dense(hiddenlayer_width, activation=self.hidden_layer_activation)
        	if hiddenlayer_index == 0:
        		nn = layer(input)
        	else:
        		nn = layer(nn)
        
        # Create output layer
        nn = Dense(self.output_shape[0], 
                   activation=self.output_layer_activation)(nn)
        
        # Create model
        self.model = Model(inputs=self.input, outputs=nn)
        
    def get_child_weights_with_partner(self, partner):
        
        # Get the weights of the agent and the partner
        my_weights = self.model.get_weights()
        partners_weights = partner.model.get_weights()
        
        # Mix those genes
        child_weights = []
        for my_layer, partners_layer in zip(my_weights, partners_weights):
            gene_crossing = np.random.randint(0,2,my_layer.shape)
            child_layer = gene_crossing * my_layer + (1-gene_crossing)*partners_layer
            child_weights.append(child_layer)
#         child.model.set_weights(child_weights)
        
        return child_weights
        
    def get_child_weights_with_partner_and_mutate(self, partner):
        
        # Get the weights of the agent and the partner
        my_weights = self.model.get_weights()
        partners_weights = partner.model.get_weights()
        
        # Mix those genes
        child_weights = []
            
        mutation_chance = self.mutation_chance
        mutation_distance = self.mutation_distance
            
        for my_layer, partners_layer in zip(my_weights, partners_weights):
            # Mixing
            gene_crossing = np.random.randint(0,2,my_layer.shape)
            child_layer = gene_crossing * my_layer + (1-gene_crossing)*partners_layer
            # Mutating
            mutation_layer = np.random.rand(*child_layer.shape) < mutation_chance
            mutation_distance_layer = mutation_distance * np.random.randn(*my_layer.shape)
            child_layer = child_layer + mutation_layer * mutation_distance_layer
            
            child_weights.append(child_layer)
        
        return child_weights
    
    def mutate(self):
        my_weights = self.model.get_weights()
        new_weights = []
        for my_layer in my_weights:
            mutation = np.random.rand(*my_layer.shape) < self.mutation_chance
            mutation_distance = self.mutation_distance * np.random.randn(*my_layer.shape)
            new_layer = my_layer + mutation * mutation_distance
            new_weights.append(new_layer)
        self.model.set_weights(new_weights)
        
    def set_weights(self, weights):
        self.model.set_weights(weights)
        
    def get_mutated_weights(self):
        my_weights = self.model.get_weights()
        new_weights = []
        for my_layer in my_weights:
            mutation = np.random.rand(*my_layer.shape) < self.mutation_chance
            mutation_distance = self.mutation_distance * np.random.randn(*my_layer.shape)
            new_layer = my_layer + mutation * mutation_distance
            new_weights.append(new_layer)
        return new_weights
        
        
    def predict(self, states):
        
        action_probabilities = self.model.predict(states)
        return action_probabilities

class DataObject():
    
    counter = 0
    start_time = int(time.time())
    
    def __init__(self, info):
        self.id = '{}_{}'.format(DataObject.start_time,DataObject.counter)
        self.parents = info['parents']
        self.fitness_value = info['fitness_value']
        self.mutated=info['mutated']
        self.cloned=info['cloned']
        
        DataObject.counter += 1
        