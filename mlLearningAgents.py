# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import collections
import random
import math
from pacman import Directions, GameState, Actions
from pacman_utils.game import Agent
from pacman_utils import util
from collections import deque

#Reference: https://github.com/mandichen/Pacman_Qlearning

class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    # Breadth First Search to find the distance to the nearest food and nearest ghost from a given position
    def bfs(self, state, start):
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        food = state.getFood()
        movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]  #positions of the neighbours
        #queue to store the neighbours and distance
        queue = deque([(start, 0)])
        visited = set()     #to keep track of visited positions
        visited.add(start)
        nearest_food_distance = None
        nearest_ghost_distance = None

        while queue:
            current, distance = queue.popleft()
            x, y = current

            #If current position is in food and nearest_food_distance is None, then we have found the nearest_food_distance
            if nearest_food_distance is None:
                if food[current[0]][current[1]]:
                    nearest_food_distance = distance

            #If the current position is in ghosts and nearest_ghost_distance is None, then we have found the nearest_ghost_distance
            if nearest_ghost_distance is None:
                if current in ghosts:
                    nearest_ghost_distance = distance

            #Check whether we have found the nearest_food_distance and nearest_ghost_distance
            if (nearest_food_distance is not None) and (nearest_ghost_distance is not None):
                return (nearest_food_distance, nearest_ghost_distance)

            for dx, dy in movements:
                nx, ny = x + dx, y + dy     #next neighbour
                if 0 <= nx < food.width and 0 <= ny < food.height and (nx, ny) not in visited: #Check whether the next neighbour does not lie outside the grid and is not visited
                    if (nx, ny) not in walls:   #Check for walls
                        queue.append(((nx, ny), distance + 1))  #Add next neighbour to queue
                        visited.add((nx, ny))  #mark visited

        return (None, None)

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        pacman = state.getPacmanPosition()
        legalActions = state.getLegalActions()

        #Mapping direction to indices
        DirToIndex = {Directions.NORTH: [0, 1], Directions.SOUTH: [0, -1], Directions.WEST:[-1, 0], Directions.EAST:[1, 0]}

        #Next position of pacman
        next_pos = [0,0]
        #feature vector for the current state
        feature = []
        foodInAllDirections = []
        ghostInAllDirections = []

        #Creating a feature vector of minimum distances to food and ghost in all four directions(N, S, E, W) from the current state
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            #check whether action is legal. If yes, then get next position
            if action in legalActions:
                next_pos[0] = DirToIndex[action][0] + pacman[0]
                next_pos[1] = DirToIndex[action][1] + pacman[1]

            #If current action is not legal, then pacman stays on same position
            else:
                next_pos[0] = pacman[0]
                next_pos[1] = pacman[1]

            #get the minimum distances to food and ghosts using BFS after considering an action
            (nearest_food_distance, nearest_ghost_distance) = self.bfs(state, (next_pos[0], next_pos[1]))

            #adding the distances to the feature vector
            feature.append(nearest_food_distance)
            feature.append(nearest_ghost_distance)

        #feature vector
        self.state = tuple(feature)


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.2,
                 gamma: float = 0.8,
                 maxAttempts: int = 6,
                 numTraining: int = 2000):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        #How many times to try each action in each state
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # Q-table
        self.Q_table = collections.defaultdict(lambda: 0)
        # Count of number of times the action has been taken in given state
        self.counts = collections.defaultdict(lambda: 0)
        # current score
        self.score = 0
        # last state
        self.lastState = []
        # last action
        self.lastAction = []

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        # reward = score of current state - score of previous state
        return endState.getScore() - startState.getScore()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        # get Q-value of a given state and action
        return self.Q_table[(state.state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        # return maximum Q-value for the possible actions
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        max_q_value = max([self.getQValue(state, action) for action in actions])
        return max_q_value

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        # Q-value of current state
        current_q_value = self.getQValue(state, action)

        # Maximum Q-value of the next state
        max_future_q_value = self.maxQValue(nextState)

        # Q-value update rule: Q(s,a) = Q(s,a) + alpha * (reward + gamma * (action with max Q-value of next state) - Q(s,a))
        new_q_value = current_q_value + self.alpha * (reward + self.gamma * max_future_q_value - current_q_value)
        self.Q_table[(state.state, action)] = new_q_value

        # Incrementing the count of state and action
        self.updateCount(state, action)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        #Updating the count for number of times the action is taken in given state
        if (state.state, action) not in self.counts:
            self.counts[(state.state, action)] = 0
        self.counts[(state.state, action)] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        #Get the count for number of times the action is taken in given state
        return self.counts[(state.state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
        #Count based exploration function mentioned in the lectures slides
        #If the count is less than maxAttempts then return the best optimal reward. Found 508.99 the best reward empirically.
        if counts < self.getMaxAttempts():
            return 508.99
        #If not then return the current utility value
        else:
            return utility

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """

        #Get the feature vector for the current state
        stateFeatures = GameStateFeatures(state)

        #get legal actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        #Check whether there is last state and calculate the reward and Q-value. Update the last state and action
        if len(self.lastState) > 0:
            reward = self.computeReward(self.lastState[-1], state)
            last_state = GameStateFeatures(self.lastState[-1])
            last_action = self.lastAction[-1]
            self.learn(last_state, last_action, reward, stateFeatures)

        # With epsilon performing a random action
        if util.flipCoin(self.epsilon):
            action = random.choice(legal)

        # With (1 - epsilon) performing action based on count based exploration function
        else:
            possibleActions = util.Counter()
            #get exploration value for each legal action
            for action in legal:
                possibleActions[action] = self.explorationFn(self.getQValue(stateFeatures, action), self.getCount(stateFeatures, action))
            #select the action with the maximum exploration value
            action = possibleActions.argMax()

        # Score of the last state
        self.score = state.getScore()
        # adding current state to last state
        self.lastState.append(state)
        # adding current action to last action
        self.lastAction.append(action)

        return action


    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        #Updating reward
        reward = self.computeReward(self.lastState[-1], state)
        #Updating Q-value
        self.learn(GameStateFeatures(self.lastState[-1]), self.lastAction[-1], reward, GameStateFeatures(state))

        #resetting last action, last state and score
        self.lastState = []
        self.lastAction = []
        self.score = 0
        # decaying epsilon
        ep = 1 - self.getEpisodesSoFar()*1.0/self.getNumTraining()
        self.setEpsilon(ep*0.1)

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)

