import gym
import pygame
import cv2
import numpy as np
from gym.utils import play
import uuid
from PIL import Image
import pickle


def crop(img, coords, saved_location=None):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.fromarray(img)
    cropped_image = image_obj.crop(coords)
    if saved_location is not None:
        cropped_image.save(saved_location)
        cropped_image.show()
    else:
        return np.asarray(cropped_image)


def callback(obs_t, obs_tp1, action, rew, done, info):
    global previous
    global data
    if obs_t is not previous:
        if action is 2 or action is 4 :
            action = 1
        if action is 3 or action is 5 :
            action = 2
        data.append((crop(cv2.cvtColor(obs_t, cv2.COLOR_BGR2GRAY), (0, 35, 160, 195)), action))
        #print(data)
        #print(len(data))
    previous = obs_t
    # print(obs_t.shape)


if __name__ == '__main__':
    data = []
    previous = 0
    env = gym.make("Pong-v0")
    play.play(env, zoom=3, callback=callback, fps=8)
    '''
    for i_episode in range(20):
        observation = env.reset()
        for t in range(1000):
            env.render()
            action = env.action_space.sample()
            #print(action)
            observation, reward, done, info = env.step(action)
            callback(observation, observation, action, reward, done, info)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break'''
    res = int(input("Le run est bon ? 1 - Oui / 2 - Non"))
    if res is 1:
        with open('save/run-' + str(uuid.uuid4()) + '.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
