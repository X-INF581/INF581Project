# Supply chain (INF581Project)

### Create the virtual env(recommended) & install requirements
```shell
$ virtualenv -p python3.6 myenv
$ source myenv/bin/activate
(myenv)$ pip install -r requirements.txt
```

### The environement

```python
from environment import Simple

env = Simple(number_of_cities=3, number_of_articles=3, capacity=[9., 13., 30.])
print(env.action_space)
print(env.observation_space)
```
```shell
# Output
range(0, 64)
ObsSpace(0.0, [9.0, 13.0, 30.0], (3, 3))
```

```python
s = env.reset()
a = 32
print(env.action_from_id(a))
next_s, r = env.step(a)
```
```shell
# Output
[[1. 0. 0.]
 [1. 0. 0.]
 [0. 0. 1.]]

[[1.33333333 1.         0.66666667]
 [0.92307692 0.92307692 0.92307692]
 [1.         1.         1.        ]]

Reward:  -1.1000000000000005
```

### Agents
#### DQN & A2C agents:
```shell
(myenv)$ python3 dql_agent.py
```

```shell
(myenv)$ python3 a2c_agent.py
```