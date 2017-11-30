# agents.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import distanceCalculator
import random, time, util
from game import Directions
import keyboardAgents
import game
from util import nearestPoint
import copy

gWeights = util.Counter()
gWeights['ourFood'] = 100.0
gWeights['theirFood'] = -100.0
gWeights['score'] = 100.0
gWeights['ourOff'] = 100.0
gWeights['ourDef'] = -100.0
gWeights['theirOff'] = 0.0
gWeights['theirDef'] = 0.0
gWeights['distToHome'] = 10.0
gWeights['ourOffRatio'] = 0.0
gWeights['theirOffRatio'] = 0.0
gWeights['bias'] = 10.0

#############
# FACTORIES #
#############

NUM_KEYBOARD_AGENTS = 0
class DeepAgentFactory(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, isRed):
    AgentFactory.__init__(self, isRed)
    gWeights = util.Counter()

  def getAgent(self, index):
    return self.choose('deep', index)

  def choose(self, agentStr, index):
    if agentStr == 'keys':
      global NUM_KEYBOARD_AGENTS
      NUM_KEYBOARD_AGENTS += 1
      if NUM_KEYBOARD_AGENTS == 1:
        return keyboardAgents.KeyboardAgent(index)
      elif NUM_KEYBOARD_AGENTS == 2:
        return keyboardAgents.KeyboardAgent2(index)
      else:
        raise Exception('Max of two keyboard agents supported')
    elif agentStr == 'deep':
      return DeepAgent(index)
    else:
      raise Exception("No staff agent identified by " + agentStr)

##########
# Agents #
##########

class DeepAgent(CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)

        # state -> action -> q-value
        self.qValues = {}

        self.discount = 0.5
        self.alpha = 0.01
        self.epsilon = 0.0
        self.score = 0

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        weights = self.getWeights()
        features = self.getFeatures(state, action)
        return weights * features

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # compute maximum q-value
        maximum = -float('inf')
        for action in state.getLegalActions(self.index):
            q = self.getQValue(state, action)
            if maximum < q:
                maximum = q
        if maximum == -float('inf'):
            return 0.0
        return maximum

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # compute action that gives maximum q-value
        maximum = -float('inf')
        maxAction = None
        for action in state.getLegalActions(self.index):
            q = self.getQValue(state, action)
            if maximum < q:
                maximum = q
                maxAction = action
        return maxAction

    def chooseAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """

        # pick action - random or greedy
        legalActions = state.getLegalActions(self.index)
        if len(legalActions) == 0:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        action = self.computeActionFromQValues(state)

        successor = self.getSuccessor(state, action)

        nextScore = self.getScore(successor)
        if self.score != nextScore:
            self.update(state, action, successor, nextScore - self.score)
            self.score = nextScore

        return action

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getWeights(self):
        global gWeights
        return gWeights

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        global gWeights
        print 'update being called'
        print gWeights

        weights = copy.deepcopy(self.getWeights())
        features = self.getFeatures(state, action)
        for i in features:
            weights[i] += \
                (self.alpha * features[i] * (reward + (self.discount * (self.computeValueFromQValues(nextState))) - self.getQValue(state, action)))
        gWeights = weights

        print gWeights

    def getFeatures(self, gameState, action):
        features = util.Counter()

        # first order features
        ourFood = self.getFood(gameState).asList()
        theirFood = self.getFoodYouAreDefending(gameState).asList()
        OUR = self.getCapsules(gameState)
        THEIR = self.getCapsulesYouAreDefending(gameState)
        squad = [s for s in self.getTeam(gameState) if s != self.index]
        them = self.getOpponents(gameState)
        score = self.getScore(gameState)
        succ = self.getSuccessor(gameState, action)
        pos = succ.getAgentState(self.index).getPosition()
        ourOff = len([oo for oo in self.getTeam(succ) if succ.getAgentState(oo).isPacman])
        ourDef = len(self.getTeam(succ)) - ourOff
        theirOff = len([to for to in self.getOpponents(succ) if succ.getAgentState(to).isPacman])
        theirDef = len(self.getTeam(succ)) - theirOff
        # our/their recently eaten (need persistent state)
        squadPos = [succ.getAgentState(sp).getPosition() for sp in self.getTeam(gameState)]
        # theirPos (need bayesian inference or particle filtering)

        # second order features
        distToHomie = self.getMazeDistance(pos, succ.getAgentState(squad[0]).getPosition())
        ourOffRatio = len(theirFood) / (ourOff + 1)
        theirOffRatio = len(ourFood) / (theirOff + 1)

        features['ourFood'] = len(ourFood)
        features['theirFood'] = len(theirFood)
        features['score'] = score
        features['ourOff'] = ourOff
        features['ourDef'] = ourDef
        features['theirOff'] = theirOff
        features['theirDef'] = theirDef
        features['distToHome'] = distToHomie
        features['ourOffRatio'] = ourOffRatio
        features['theirOffRatio'] = theirOffRatio
        features['bias'] = 1.0

        return features

    def getSuccessor(self, state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = state.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor
