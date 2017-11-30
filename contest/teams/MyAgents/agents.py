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
import random,util,math

#############
# FACTORIES #
#############

NUM_KEYBOARD_AGENTS = 0
class DeepAgentFactory(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, isRed):
    AgentFactory.__init__(self, isRed)

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

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()

    # variables shared amongst the different positions
    myFood = self.getFood(gameState).asList()
    opFood = self.getFoodYouAreDefending(gameState).asList()
    myCapsule = self.getCapsules(gameState)
    opCapsule = self.getCapsulesYouAreDefending(gameState)
    my = self.getTeam(gameState)
    op = self.getOpponents(gameState)
    score = self.getScore(gameState)

    values = [self.evaluate(gameState, a, myFood, opFood, myCapsule, opCapsule) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action, myFood, opFood, myCapsule, opCapsule):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action, myFood, opFood, myCapsule, opCapsule)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action, myFood, opFood, myCapsule, opCapsule):
    features = util.Counter()

    successor = self.getSuccessor(gameState, action)
    myPos = successor.getAgentState(self.index).getPosition()
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

    # the score achieved
    features['score'] = self.getScore(successor)

    # number of food pellets left
    features['numOfFood'] = len(myFood)

    # distance to nearest food pellet
    minDist = min([self.getMazeDistance(myPos, food) for food in myFood])
    features['minDistToFood'] = minDist

    # distance to nearest capsule
    minDist = 0
    if len(myCapsule) > 0:
        minDist = min([self.getMazeDistance(myPos, capsule) for capsule in myCapsule])
    features['minDistToCapsule'] = minDist

    # number of ghosts
    features['numOfGhosts'] = len(ghosts)

    # distance to nearest ghost
    minDist = 0
    if len(ghosts) > 0:
      minDist = min([self.getMazeDistance(myPos, a.getPosition()) for a in ghosts])
    features['minDistToGhost'] = minDist

    return features

  def getWeights(self, gameState, action):
    return {
        'score'                 : 100,
        'numOfFood'             : -10,
        'minDistToFood'         : -50,
        'minDistToCapsule'      : -2,
        'numOfGhost'            : 5,
        'minDistToGhost'        : 20,
        }

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action, myFood, opFood, myCapsule, opCapsule):
    features = util.Counter()

    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # number of food pellets left
    features['numOfFood'] = len(opFood)

    # distance to nearest food pellet
    #minDist = min([self.getMazeDistance(myPos, food) for food in myFood])
    #features['minDistToFood'] = minDist

    # distance to nearest capsule
    #minDist = 0
    #if len(myCapsule) > 0:
    #    minDist = min([self.getMazeDistance(myPos, capsule) for capsule in myCapsule])
    #features['minDistToCapsule'] = minDist

    # number of invaders
    features['numOfInvaders'] = len(invaders)

    # distance to nearest invader
    minDist = 0
    if len(invaders) > 0:
      minDist = min([self.getMazeDistance(myPos, a.getPosition()) for a in invaders])
    features['minDistToInvader'] = minDist

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {
        'onDefense'         : 100,
        'numOfFood'         : 10,
        'numOfInvaders'     : -500,
        'minDistToInvader'  : -100,
        'stop'              : -10,
        'reverse'           : -2
        }

class DAgent(ReflexCaptureAgent):

  def __init__(self, index, timeForComputing=.1):
    ReflexCaptureAgent.__init__(self, index, timeForComputing)
    self.score = 0.0
    self.Q = util.Counter()
    self.alpha = 0.1
    sef.discount = 0.9

  def chooseAction(self, state):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(state, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getQValue(self, gameState, action):
    return self.getWeights(gameState, action) * self.getFeatures(gameState, action)

  def update(self, gameState, action, nextGameState, reward):
    weights = copy.deepcopy(self.getWeights(gameState, action))
    features = self.getFeatures(gameState, action)
    for i in features:
        weights[i] += \
            (self.alpha * features[i] * (reward + (self.discount * (self.computeValueFromQValues(nextGameState))) - self.getQValue(gameState, action)))
    self.weights = weights

  def getFeatures(self, gameState, action, myFood, opFood, myCapsule, opCapsule):
    features = util.Counter()

    # bias
    features['bias'] = 1.0

    succ = self.getSuccessor(gameState, action)
    pos = succ.getAgentState(self.index).getPosition()
    team = [t for t in self.getTeam(gameState) if t != self.index]
    enemies = [succ.getAgentState(i) for i in self.getOpponents(succ)]
    invaders = [a for a in enemies if a.isPacman]

    # first order
    numMyFood = len(myFood)
    numOpFood = len(opFood)
    opOff = len(invaders)
    opDef = len(self.getOpponents(succ)) - opOff
    distToTeam = self.getMazeDistance(pos, succ.getAgentState(team[0]).getPosition())
    print distToTeam


    # second order
    # print dir(gameState)
    # print gameState.getWalls().width

    return features

  def getWeights(self, gameState, action):
    return self.weights

  def updateWeights(self, reward, successor):
    diff = reward + (0.9 * self.V(successor)) - self.Q[successor]
    for k in self.features:
        self.weights[k] += (0.1 * diff * self.features[k])

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      toReturn = successor.generateSuccessor(self.index, action)
    else:
      toReturn = successor

    nextScore = self.getScore(toReturn)
    if self.score != nextScore:
        self.updateWeights(nextScore - self.score)
        self.score = nextScore

    return toReturn











class DeepAgent(CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)

        # state -> action -> q-value
        self.qValues = {}

        self.discount = 0.9
        self.alpha = 0.1
        self.epsilon = 0.2
        self.weights = util.Counter()
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
        return self.weights

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        print 'update being called'
        print self.weights

        weights = copy.deepcopy(self.getWeights())
        features = self.getFeatures(state, action)
        for i in features:
            weights[i] += \
                (self.alpha * features[i] * (reward + (self.discount * (self.computeValueFromQValues(nextState))) - self.getQValue(state, action)))
        self.weights = weights

        print self.weights

    def getFeatures(self, gameState, action):
        self.features = util.Counter()
        self.features['bias'] = 1.0
        return self.features

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
