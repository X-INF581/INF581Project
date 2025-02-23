import numpy as np


class ObservationSpace:
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape

    def __repr__(self):
        return "ObsSpace({}, {}, {})".format(self.low, self.high, self.shape)


class Simple():
    def __init__(
        self,
        number_of_cities,
        number_of_articles,
        capacity,
        reward_same_city=2,
        reward_other_city=1,
        reward_not_found=-1,
        reward_overflow=-1,
        reward_holding=-0.1,
        probabilities=None
    ):
        
        self.number_of_cities = number_of_cities
        self.number_of_articles = number_of_articles
        if isinstance(capacity, list):
            if len(capacity) != number_of_cities:
                raise ValueError("Capacities need to be equal to the number of cities")
            self.capacity = np.array(capacity).reshape((-1, 1))
        else:
            self.capacity = capacity
        if probabilities:
            self.probabilities = probabilities
        else:
            self.probabilities = np.random.random((number_of_cities,number_of_articles))
            self.probabilities /= np.expand_dims(np.sum(self.probabilities,axis=1),axis=1)
        self.limites = np.ones(number_of_cities)*capacity
        self.reward_not_found = reward_not_found
        self.reward_other_city = reward_other_city
        self.reward_overflow = reward_overflow
        self.reward_same_city = reward_same_city
        self.reward_holding = reward_holding
        self.warehouses = (self.capacity//self.number_of_articles)*np.ones((self.number_of_cities,self.number_of_articles))
        # adding the "Put-No-Article" action to the action space
        self.action_space = range(0, (number_of_articles+1)**number_of_cities)
        self.observation_space = ObservationSpace(low=0., high=capacity, shape=(number_of_cities, number_of_articles))
        # Track the current performance based on the deliveries
        self.total_orders = 0
        self.good_delivery = 0
        self.medium_delivery = 0
        self.missing_delivery = 0
        

    def action_from_id(self, n):
        if (n not in self.action_space):
            raise ValueError("{} is not in the action space {}".format(n, self.action_space))
        res = np.zeros((self.number_of_cities,self.number_of_articles))
        for i in range(self.number_of_cities):
            idx = n % (self.number_of_articles + 1)
            # adding the article_i in the warehouse of the city 
            # the else part means Put nothing (No action) 
            if idx < self.number_of_articles:
                res[i][idx] = 1
            n = n // (self.number_of_articles + 1)
        return res
    
    
    def id_from_actions(self, actions):
        a = 0
        for i in range(0, len(actions)):
            a += actions[i] * (self.number_of_articles + 1)**i
        assert (a in self.action_space)
        return a
    

    def reset(self):
        self.total_orders = 0
        self.good_delivery = 0
        self.medium_delivery = 0
        self.missing_delivery = 0
        self.warehouses = (self.capacity//self.number_of_articles)*np.ones((self.number_of_cities,self.number_of_articles))
        return self.number_of_articles*self.warehouses/self.capacity
    

    def step(self,action):
        reward = 0
        self.warehouses += self.action_from_id(action)
        filling = np.sum(self.warehouses,axis=1)
        reward += self.reward_overflow*np.sum(filling>self.limites)
        self.warehouses = np.minimum(self.warehouses,self.limites.reshape((-1, 1)))
        orders = []
        deliveries = np.zeros((self.number_of_cities,self.number_of_articles))
        for i in range(self.number_of_cities):
            self.total_orders  += 1
            order = np.random.choice(np.arange(0,self.number_of_articles),p=self.probabilities[i])
            orders.append(order)
            found_same_city = False
            found_other_city = False
            if self.warehouses[i][order]>0:
                reward+= self.reward_same_city
                self.good_delivery += 1
                deliveries[i,order] += 1
                self.warehouses[i,order] -= 1 
                found_same_city = True
            else:
                for j in range(self.number_of_cities):
                    if j != i and self.warehouses[j][order]>0:
                        reward+= self.reward_other_city
                        self.medium_delivery += 1
                        deliveries[j,order] += 1
                        self.warehouses[j,order] -= 1
                        found_other_city = True 
                        break
            if not found_same_city:
                # giving a negative reward for the 2 cases (other_city and missing delivery)
                reward += self.reward_not_found

            if (not found_same_city and not found_other_city):
                self.missing_delivery += 1

        orders = np.eye(self.number_of_articles)[orders]
        reward += self.reward_holding*np.sum(self.warehouses)
        return self.number_of_articles*self.warehouses/self.capacity,reward


def main():
    simple = Simple(number_of_cities=3, number_of_articles=3, capacity=[9., 13., 30.])
    print(simple.limites)
    print(simple.action_space)
    print(simple.observation_space)
    simple.reset()
    print(simple.warehouses)
    print(simple.step(32))
    ########################################
    print()
    actions = [1, 0, 3]
    a = simple.id_from_actions(actions)
    print(a)
    print(simple.action_from_id(a))
    
if __name__=="__main__":
    main() 