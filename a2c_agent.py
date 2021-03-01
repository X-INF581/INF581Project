#! /usr/bin/env python3
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input
from environment import Simple
import matplotlib.pyplot as plt
plt.style.use("ggplot")


class A2C:
	def __init__(self, obs_shape, n_action, gamma=0.99):
		self.obs_shape = obs_shape
		self.n_action = n_action
		self.gamma = gamma
		g = tf.Graph()
		with g.as_default():
			self.states = tf.compat.v1.placeholder(tf.float32, shape=(None, self.obs_shape), name="s")
			self.next_states = tf.compat.v1.placeholder(tf.float32, shape=(None, self.obs_shape), name="next_s")
			self.reward = tf.compat.v1.placeholder(tf.float32, name="reward")
			self.actions = tf.compat.v1.placeholder(tf.int32, name="actions")
			self.common_linear_layer = Dense(units=100, activation="linear")
			loss = self._get_loss()
			self.optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)
			init = tf.compat.v1.global_variables_initializer()
		self.sess = tf.compat.v1.Session(graph=g)
		self.sess.run(init)
	
	def _get_loss(self):
		x = self.common_linear_layer(self.states)
		next_x = self.common_linear_layer(self.next_states)
		logits = Dense(units=self.n_action, activation="linear")(x)
		self.actor_policy = tf.nn.softmax(logits)
		log_policy = tf.nn.log_softmax(logits)
		state_value = Dense(units=1, activation="linear")(x)
		next_state_value = Dense(units=1, activation="linear")(next_x)
		target_value = self.reward + self.gamma * next_state_value
		critic_loss = tf.reduce_mean((tf.stop_gradient(target_value) - state_value)**2)
		advantage = target_value - state_value
		J = tf.reduce_mean(
			tf.reduce_sum(log_policy * tf.one_hot(self.actions, self.n_action), axis=1) * tf.stop_gradient(advantage)
		)
		entropy = - tf.reduce_mean(tf.reduce_sum(log_policy * self.actor_policy, axis=1))
		loss = - J + 0.5 * critic_loss - 0.001 * entropy
		return loss
	
	def train(self, s, next_s, reward, action):
		self.sess.run(
			self.optimizer,
			feed_dict={
				self.states: [s],
				self.next_states: [next_s],
				self.reward: reward,
				self.actions: action,
			}			
		)
	
	def get_action_probs(self, s):
		probs = self.sess.run(
			self.actor_policy,
			feed_dict={
				self.states: [s]
			}
		)
		return probs[0]


def generate_trajectory(agent, env, t_max=100):
	s = env.reset()
	total_rewards = 0
	s = s.flatten()
	for _ in range(t_max):
		proba_dist = agent.get_action_probs(s)
		action = np.random.choice(env.action_space, p=proba_dist)
		next_s, r = env.step(action)
		next_s  = next_s.flatten()
		r /= 4
		total_rewards += r
		agent.train(s, next_s, r, action)
		s = next_s
	return total_rewards
		

def main():
	nbr_articles = 3
	nbr_cities = 3
	env = Simple(number_of_cities=nbr_cities, number_of_articles=nbr_articles, capacity=30.)
	n_action = (nbr_articles+1)**nbr_cities
	obs_shape = nbr_articles * nbr_cities
	agent = A2C(
		obs_shape=obs_shape,
		n_action=n_action
	)
	means_rewards = []
	epochs = 50
	steps = 20
	for epoch in range(0, epochs):
		sum_r = 0
		for s in range(0, steps):
			sum_r += generate_trajectory(agent, env)
			good_rate = (env.good_delivery / env.total_orders) * 100
			meduim_rate = (env.meduim_delivery / env.total_orders) * 100
			missing_rate = (env.missing_delivery / env.total_orders) * 100
			print("Episode {}: Total-orders={}, good-deliveries={:.2f} %, meduim-deliveries={:.2f} %, missing-deliveries={:.2f} %"\
				.format(s, env.total_orders, good_rate, meduim_rate, missing_rate)
				)
		mean_r = sum_r / steps 
		means_rewards.append(mean_r)
		print("Epoch: {} | mean reward: {}".format(epoch, mean_r))
		print()
	plt.plot(means_rewards)
	plt.xlabel("Epochs")
	plt.ylabel("mean rewards")
	plt.savefig("a2c.png")
	plt.show()
	

if __name__ == "__main__":
	main()
