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

    values = [self.evaluate(gameState, a) for a in actions]
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

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
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

class DeepAgent(ReflexCaptureAgent):

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

    # bias
    features['bias'] = 1.0

    return features

  def getWeights(self, gameState, action):

    if not self.weights:
        self.weights = util.Counter()

    return self.weights

  def updateWeights(self, reward):
    for k in self.features:
        self.features[k] *= (0.1 * ())
    self.weights = self.weights +

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      toReturn = successor.generateSuccessor(self.index, action)
    else:
      toReturn = successor

    if not self.score:
        self.score = 0

    curScore = self.getScore(toReturn)
    if self.score != curScore:
        self.updateWeights(curScore - self.score)
        self.score = curScore

    return toReturn
