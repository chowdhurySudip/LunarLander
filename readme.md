# Solving Lunar Lander Problem Using DQN

In recent years, deep reinforcement learning has seen major breakthroughs in solving the real world challenges in various domains. Deep Q-Network (DQN), implemented by Minh [1], is one of those early breakthroughs that achieved state-of-the-art results in different Atari Games without any handcrafted features or change in learning algorithm. This article demonstrates the effectiveness of DQN in solving OpenAI gym’sLunar Lander problem. Similar to Minh, I also used ANNs to approximate the state-action value function. Moreover, I used hyperparameter tuning to identify the best set of parameters that solves the problem with least number of episodes. During this experimentation, I found that the exploration strategy, architecture of the ANN and discount rate has the most impact on the agent’s training process.

## Getting Started

For quick experiments, you can use [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb) using `LunarLander_v2_dqn.ipynb`. You can also run the experiments locally.

### Requirements

#### For running locally
-   Install numpy
-   Install matplotlib
-   Install tqdm
-   Install gym
-   Install pytorch
-   Install pyvirtualdisplay - sudo apt-get install -y xvfb python-opengl, pip install pyvirtualdisplay
-   Install tensorflow_docs - pip install git+https://github.com/tensorflow/docs


#### For running in Colab
-   !pip3 install box2d-py
-   Install pyvirtualdisplay - sudo apt-get install -y xvfb python-opengl, pip install pyvirtualdisplay
-   Install tensorflow_docs - pip install git+https://github.com/tensorflow/docs


### Requirements
- Mnih, V. et al. “Human-level control through deep reinforcement learning.” Nature 518 (2015): 529-533.
- Brockman, G. et al. “OpenAI Gym.” ArXiv abs/1606.01540 (2016): n. pag.
- Sutton and Barto, Reinforcement Learning: An Introduction, 2nd edition http://incompleteideas.net/sutton/book/code/code2nd.html
