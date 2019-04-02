"""
Nothing for the moment, juste load data
"""

from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
import gym
import cv2
from PongIA import crop
import keras
from keras import layers
from keras import utils

def load():
    """
    load data from run
    :return: an array of 2-tuple,
    the first element represent the image
    with a matrix of 160x160 and the second
    list is the action
    """
    mypath = 'save/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    loadData = []
    for f in onlyfiles:
        with open(mypath + f, 'rb') as pick:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            loadData = loadData + pickle.load(pick)
    return loadData


def interpretOutput(action):
    #Convert the neural network's output to an action
    if int(action[0][0]) == 1 : 
        print('ACTION 0')
        return 0
    elif int(action[0][1]) == 1:
        print('ACTION 2')
        return 2
    elif int(action[0][2]) == 1: 
        print('ACTION 3')
        return 3
    return 0

if __name__ == '__main__':
    data = load()

    x_train = [ seq[0] for seq in data ]
    y_train = [ seq[1] for seq in data ]

    x_train = np.array(x_train)
    y_train = keras.utils.to_categorical(y_train, 3)

    #Use small simple to test our model, it's not useful for the problem
    test = data[:199]
    x_test = [ seq[0] for seq in test ]
    y_test = [ seq[1] for seq in test ]
    x_test = np.array(x_test)
    y_test = keras.utils.to_categorical(y_test, 3)


    #Neural network definition
    inputs = keras.Input(shape=(160,160))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(8, activation='relu')(x)
    outputs = layers.Dense(3, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.summary() #Print the network shape
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    
    model.fit(x_train, y_train, epochs=10, validation_split=0.33)
    
    #Score to evaluate our model
    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)

    #Play the game with our neural network
    env = gym.make('Pong-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(1000):
            env.render()
            observation = crop(cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY), (0, 35, 160, 195))
            observation = np.expand_dims(observation, axis=0)
            action = model.predict(observation)
            action = interpretOutput(action)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()