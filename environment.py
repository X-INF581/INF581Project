import numpy as np
class Simple():
    
    def __init__(self,number_of_cities=3,number_of_articles=3,capacity=3,reward_same_city=2,reward_other_city=1,reward_not_found=-1,reward_overflow=-1):
        self.number_of_cities = number_of_cities
        self.number_of_articles = number_of_articles
        self.probabilities = np.random.random((number_of_cities,number_of_articles))
        self.probabilities /= np.expand_dims(np.sum(self.probabilities,axis=1),axis=1)
        self.warehouses = np.zeros((number_of_cities,number_of_articles))
        self.limites = np.ones(number_of_cities)*capacity
        self.reward_not_found = reward_not_found
        self.reward_other_city = reward_other_city
        self.reward_overflow = reward_overflow
        self.reward_same_city = reward_same_city

    def reset(self):
        self.warehouses = np.zeros((self.number_of_cities,self.number_of_articles))
    def step(self,action):
        reward = 0 
        self.warehouses += action
        filling = np.sum(self.warehouses,axis=1)
        reward += self.reward_overflow*np.sum(filling>self.limites)
        orders = []
        deliveries = np.zeros((self.number_of_cities,self.number_of_articles))
        for i in range(self.number_of_cities):
            order = np.random.choice(np.arange(0,self.number_of_articles),p=self.probabilities[i])
            orders.append(order)
            found = False
            if self.warehouses[i][order]>0:
                reward+= self.reward_same_city
                deliveries[i,order] += 1
                self.warehouses[i,order] -= 1 
                found = True
            else:
                for j in range(self.number_of_cities):
                    if j != i and self.warehouses[j][order]>0:
                        reward+= self.reward_other_city
                        deliveries[j,order] += 1
                        self.warehouses[j,order] -= 1 
                        break
            if not found:
                reward += self.reward_not_found

                        


        orders = np.eye(self.number_of_articles)[orders]
        return (self.warehouses,orders),reward

def main():
    simple = Simple()
    #s,r = simple.step(np.random.randint(0,2,size=(3,3)))
    s,r = simple.step(np.zeros((3,3)))
    print(s,r)
if __name__=="__main__":
    main()

        