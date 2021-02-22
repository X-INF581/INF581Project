import environment
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from collections import deque
import numpy as np
import time
import random
import matplotlib.pyplot as plt
t = time.time()
import keras as keras
def create_model(number_of_cities=3,number_of_articles=3):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(number_of_articles*number_of_cities,)))
    model.add(keras.layers.Dense(number_of_articles**number_of_cities))
    model.compile('adam',loss='mse')
    return model


number_of_cities,number_of_articles = 1,5
simple = environment.Simple(capacity=15,number_of_cities=number_of_cities,number_of_articles=number_of_articles,reward_overflow=-2,reward_same_city=4)
simple.warehouses += 2

def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]))[0]

def train(env, replay_memory, model, target_model, done):
    learning_rate = 0.03 # Learning rate
    discount_factor = 0.9

    MIN_REPLAY_SIZE = 200
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64 * 3
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([encode_observation(transition[0], (number_of_cities,number_of_articles)) for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([encode_observation(transition[3], (number_of_cities,number_of_articles)) for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(encode_observation(observation, (number_of_cities,number_of_articles)))
        Y.append(current_qs)

    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

def encode_observation(observation, n_dims):
    return observation.flatten()

def main():
    rewards = []
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.1 # At a minimum, we'll always explore 1% of the time
    decay = 0.02
    print(simple.probabilities)
    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    model = create_model(number_of_articles=number_of_articles, number_of_cities=number_of_cities)
    # Target Model (updated every 100 steps)
    target_model = create_model(number_of_articles=number_of_articles,number_of_cities=number_of_cities)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=1_000)

    target_update_counter = 0

    # X = states, y = actions
    X = []
    y = []

    steps_to_update_target_model = 0
    states = []
    for episode in range(700):
        total_training_rewards = 0
        observation = simple.reset()
        done = False
        while not done:
            steps_to_update_target_model += 1
            random_number = np.random.rand()
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                action = np.random.randint(0,number_of_articles**number_of_cities)
            else:
                # Exploit best known action
                # model dims are (batch, env.observation_space.n)
                encoded = encode_observation(observation, number_of_cities)
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                predicted = model.predict(encoded_reshaped).flatten()
                action = np.argmax(predicted)

            states.append(observation)
            new_observation, reward = simple.step(action)
            reward /= 4
            replay_memory.append([observation, action, reward, new_observation, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(simple, replay_memory, model, target_model, done)

            observation = new_observation
            total_training_rewards += reward

            if steps_to_update_target_model>25:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                rewards.append(total_training_rewards)
                total_training_rewards += 1
                
                #case_tests = np.array([[0,1,1],[1,0,1],[1,1,0]])
                #print(model.predict(case_tests))
                if steps_to_update_target_model >= 20:
                    #print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                    print(observation)
                    break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    plt.plot(np.convolve(rewards,np.ones(10)/10,mode='valid'))
    plt.show()
    states = np.array(states)
    print(simple.probabilities)
    for i in range(number_of_articles):
        plt.plot(states[:,0,i])
        plt.show()
main()