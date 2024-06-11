# Q-Learning for pacman

mlLearningAgents.py file with contains QLearnAgent class that performs Q-learning with exploration. The learner includes two separate pieces of code:

1. Something that learns, adjusting utilities based on how well the learner plays the game.
2. Something that chooses how to act. This decision can be based on the utilities, but also has to make sure that the learner does enough exploring.

The only way that Pacman learns is by actually playing in a game. Initially they will play badly, but over time they do get better and better. 

Command to run the file: python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid - this will train the learner for 2000 episodes and then run it for 10 non-training episodes. 
