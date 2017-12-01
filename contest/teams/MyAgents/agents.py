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
      NUM_KEYBOARD_AGENTS += 2
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

class DeepAgent(ReflexCaptureAgent):

  # static variables
  init = False
  legalPositions = None
  numParticles = 500
  particleDict = {}

  def initialize(self, gameState):
    if DeepAgent.legalPositions is None:
      DeepAgent.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]

  def updateSharedInfo(self, gameState, myPos):
    if not DeepAgent.init:
      self.initialize(gameState)

    for opp in self.getOpponents(gameState):
      if opp not in DeepAgent.particleDict:
        self.initParticles(opp)
      self.filterParticles(opp, gameState.getAgentDistances()[opp], myPos)

  def initParticles(self, index):
    particles = []
    for i in range(DeepAgent.numParticles):
      particles.append(DeepAgent.legalPositions[i % len(DeepAgent.legalPositions)])
    DeepAgent.particleDict[index] = particles

  def filterParticles(self, index, observation, myPos):
    emissionModel = util.Counter()
    # hardcoded uniform distribution [-6, +6]
    for i in range(observation - 6, observation + 7):
      emissionModel[i] = 1.0 / 13
    beliefs = self.getBeliefDistribution(index)

    allPossible = util.Counter()
    for lp in DeepAgent.legalPositions:
      dist = util.manhattanDistance(lp, myPos)
      allPossible[lp] = emissionModel[dist] * beliefs[lp]
    allPossible.normalize()

    if allPossible.totalCount() != 0:
      newParticles = []
      for i in range(DeepAgent.numParticles):
        newParticles.append(util.sample(allPossible))
      DeepAgent.particleDict[index] = newParticles
    else:
      self.initParticles(index)

  def getBeliefDistribution(self, index):
    freq = util.Counter()
    for p in DeepAgent.particleDict[index]:
      freq[p] += 1
    freq.normalize()
    return freq

  """
  Only call this on enemy indices
  """
  def getPosition(self, index, gameState):
    if gameState.getAgentState(index).getPosition() is not None:
      return gameState.getAgentState(index).getPosition()
    else:
      return self.getBeliefDistribution(index).argMax()

  def getFeatures(self, gameState, action):

    succ = self.getSuccessor(gameState, action)
    myPos = succ.getAgentState(self.index).getPosition()
    width = gameState.getWalls().width
    height = gameState.getWalls().height
    xBorder = gameState.getWalls().width / 2
    squad = [s for s in self.getTeam(gameState) if s != self.index]
    ourFood = self.getFoodYouAreDefending(gameState).asList()
    theirOff = [to for to in self.getOpponents(succ) if succ.getAgentState(to).isPacman]

    self.updateSharedInfo(gameState, myPos)

    ### features
    ## offense
    # len(theirFood)
    # distToTheirFoodCenter
    # minDistToFood
    # minDistToCapsule
    # minDistToGhost
    # scaredTime
    # recentlyEaten
    # theirDef
    # ourOff
    ## defense
    # len(ourFood)
    # theirOff
    # ourDef
    # minDistToInvader
    # distToOurFoodBorderCenter
    ## general
    # minDistToHomie
    # score
    # numAgents
    # stop
    # reverse
    # bias
    # ??? combine ourOffRatio and theirOffRatio ???

    # defense
    minDistToInvader = 0
    if len(theirOff) != 0:
      minDist = width
      for to in theirOff:
        dist = self.getMazeDistance(myPos, self.getPosition(to, gameState))
        minDist = min(dist, minDist)
      minDistToInvader = minDist

    distToFoodBorderCenter = 0
    foodCenter = [0, 0]
    for of in ourFood:
      foodCenter = list(sum(c) for c in zip(foodCenter, of))
    foodCenter = list(c / len(ourFood) for c in foodCenter)
    foodCenter[0] = (foodCenter[0] + xBorder) / 2
    distToFoodBorderCenter = self.getMazeDistance(myPos, tuple(foodCenter))

    minDistToHomie = width + height
    for s in squad:
      dist = self.getMazeDistance(myPos, gameState.getAgentState(s).getPosition())
      minDistToHomie = min(dist, minDistToHomie)

    # print yDispersionDiff
    # # first order features
    # ourFood = self.getFood(gameState).asList()
    # theirFood = self.getFoodYouAreDefending(gameState).asList()
    # OUR = self.getCapsules(gameState)
    # THEIR = self.getCapsulesYouAreDefending(gameState)
    # squad = [s for s in self.getTeam(gameState) if s != self.index]
    # them = self.getOpponents(gameState)
    # score = self.getScore(gameState)
    # succ = self.getSuccessor(gameState, action)
    # pos = succ.getAgentState(self.index).getPosition()
    # ourOff = len([oo for oo in self.getTeam(succ) if succ.getAgentState(oo).isPacman])
    # ourDef = len(self.getTeam(succ)) - ourOff
    # theirOff = len([to for to in self.getOpponents(succ) if succ.getAgentState(to).isPacman])
    # theirDef = len(self.getTeam(succ)) - theirOff
    # # our/their recently eaten (need persistent state)
    # squadPos = [succ.getAgentState(sp).getPosition() for sp in self.getTeam(gameState)]
    # # theirPos (need bayesian inference or particle filtering)
    #
    # # second order features
    # distToHomie = self.getMazeDistance(pos, succ.getAgentState(squad[0]).getPosition())
    # ourOffRatio = len(theirFood) / (ourOff + 1)
    # theirOffRatio = len(ourFood) / (theirOff + 1)

    features = util.Counter()
    features['minDistToInvader'] = minDistToInvader
    features['distToFoodBorderCenter'] = distToFoodBorderCenter
    features['minDistToHomie'] = minDistToHomie

    # features['ourFood'] = len(ourFood)
    # features['theirFood'] = len(theirFood)
    # features['score'] = score
    # features['ourOff'] = ourOff
    # features['ourDef'] = ourDef
    # features['theirOff'] = theirOff
    # features['theirDef'] = theirDef
    # features['distToHome'] = distToHomie
    # features['ourOffRatio'] = ourOffRatio
    # features['theirOffRatio'] = theirOffRatio

    features['bias'] = 1.0
    return features

  def getWeights(self, gameState, action):
    return {
      'minDistToInvader'            : -100,
      'distToFoodBorderCenter'      : -10,
      'minDistToHomie'              : 3,
    }
