# keyboardAgents.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import Agent
from game import Directions
import random

import sys
sys.path.insert(0, 'teams/MyAgents/')
from agents import ReflexCaptureAgent, DeepAgent
import util
import pickle

# UPDATE FOR EACH NEW GAME
GAME_ID = '0'

class KeyboardAgent(ReflexCaptureAgent):
  """
  An agent controlled by the keyboard.
  """
  # NOTE: Arrow keys also work.
  WEST_KEY  = 'a'
  EAST_KEY  = 'd'
  NORTH_KEY = 'w'
  SOUTH_KEY = 's'
  STOP_KEY  = 'q'

  def __init__( self, index = 0 ):
    self.lastMove = Directions.STOP
    self.index = index
    self.keys = []
    self.befores = []
    self.afters = []

  def getAction(self, state):

    self.befores.append(self.getFeatures(state).values())

    from graphicsUtils import keys_waiting
    from graphicsUtils import keys_pressed
    keys = keys_waiting() + keys_pressed()
    if keys != []:
      self.keys = keys

    legal = state.getLegalActions(self.index)
    move = self.getMove(legal)

    if move == Directions.STOP:
      # Try to move in the same direction as before
      if self.lastMove in legal:
        move = self.lastMove

    if (self.STOP_KEY in self.keys) and Directions.STOP in legal: move = Directions.STOP

    if move not in legal:
      move = random.choice(legal)

    self.lastMove = move

    self.afters.append(self.getFeaturesAfterAction(state, move).values())
    global GAME_ID
    pickle.dump(self.befores, open('training/eric-befores-' + GAME_ID + '.pkl', 'w'))
    pickle.dump(self.afters, open('training/eric-afters-' + GAME_ID + '.pkl', 'w'))

    return move

  def getMove(self, legal):
    move = Directions.STOP
    if   (self.WEST_KEY in self.keys or 'Left' in self.keys) and Directions.WEST in legal:  move = Directions.WEST
    if   (self.EAST_KEY in self.keys or 'Right' in self.keys) and Directions.EAST in legal: move = Directions.EAST
    if   (self.NORTH_KEY in self.keys or 'Up' in self.keys) and Directions.NORTH in legal:   move = Directions.NORTH
    if   (self.SOUTH_KEY in self.keys or 'Down' in self.keys) and Directions.SOUTH in legal: move = Directions.SOUTH
    return move

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


  def getPosition(self, index, gameState):
    """
    Only call this on enemy indices.
    """
    if gameState.getAgentState(index).getPosition() is not None:
      return gameState.getAgentState(index).getPosition()
    else:
      return self.getBeliefDistribution(index).argMax()

  def getFeaturesAfterAction(self, gameState, action):
    return self.getFeatures(self.getSuccessor(gameState, action))

  def getFeatures(self, succ):
    myPos = succ.getAgentState(self.index).getPosition()
    width = succ.getWalls().width
    height = succ.getWalls().height
    xBorder = succ.getWalls().width / 2
    squad = [s for s in self.getTeam(succ) if s != self.index]
    ourFood = self.getFoodYouAreDefending(succ).asList()
    OUR = self.getCapsulesYouAreDefending(succ)
    ourOff = [oo for oo in self.getTeam(succ) if succ.getAgentState(oo).isPacman]
    ourDef = [od for od in self.getTeam(succ) if not succ.getAgentState(od).isPacman]
    theirFood = self.getFood(succ).asList()
    THEIR = self.getCapsules(succ)
    theirOff = [to for to in self.getOpponents(succ) if succ.getAgentState(to).isPacman]
    theirDef = [td for td in self.getOpponents(succ) if not succ.getAgentState(td).isPacman]

    self.updateSharedInfo(succ, myPos)

    ### features
    ## offense
    # len(theirFood)
    # distToTheirFoodCenter
    # minDistToFood
    # minDistToCapsule
    # minDistToGhost
    # scaredTime (TODO)
    # recentlyEaten (TODO)
    # ourOff
    # theirDef
    ## defense
    # len(ourFood)
    # ourDef
    # theirOff
    # minDistToInvader
    # distToOurFoodBorderCenter
    ## general
    # minDistToHomie
    # score
    # stop
    # reverse
    # bias
    # ??? combine ourOffRatio and theirOffRatio ???

    ## offense
    numTheirFood = len(theirFood)

    tfc = [0, 0]
    for tf in theirFood:
      tfc = list(sum(c) for c in zip(tfc, tf))
    tfc = list(c / len(theirFood) for c in tfc)
    if tfc not in DeepAgent.legalPositions:
      for lp in DeepAgent.legalPositions:
        if util.manhattanDistance(tfc, lp) == 2:
          tfc = lp
          break
    distToTheirFoodCenter = self.getMazeDistance(myPos, tuple(tfc))

    minDistToFood = width + height
    for tf in theirFood:
      minDistToFood = min(minDistToFood, self.getMazeDistance(myPos, tf))

    minDistToCapsule = 0 if len(THEIR) == 0 else (width + height)
    for tc in THEIR:
      minDistToCapsule = min(minDistToCapsule, self.getMazeDistance(myPos, tc))

    minDistToGhost = width + height
    for td in theirDef:
      minDistToGhost = min(minDistToGhost, self.getMazeDistance(myPos, self.getPosition(td, succ)))

    numOurOff = len(ourOff)
    numTheirDef = len(theirDef)

    ## defense
    numOurFood = len(ourFood)
    numTheirOff = len(theirOff)
    numOurDef = len(ourDef)

    minDistToInvader = 0 if len(theirOff) == 0 else (width + height)
    for to in theirOff:
      minDistToInvader = min(minDistToInvader, self.getMazeDistance(myPos, self.getPosition(to, succ)))

    ofc = [0, 0]
    for of in ourFood:
      ofc = list(sum(c) for c in zip(ofc, of))
    ofc = list(c / len(ourFood) for c in ofc)
    ofc[0] = (ofc[0] + xBorder) / 2
    if ofc not in DeepAgent.legalPositions:
      for lp in DeepAgent.legalPositions:
        if util.manhattanDistance(ofc, lp) == 1:
          ofc = lp
          break
    distToOurFoodBorderCenter = self.getMazeDistance(myPos, tuple(ofc))

    ## general
    minDistToHomie = width + height
    for s in squad:
      dist = self.getMazeDistance(myPos, succ.getAgentState(s).getPosition())
      minDistToHomie = min(dist, minDistToHomie)

    score = self.getScore(succ)
    stop = 1.0
    reverse = 1.0
    bias = 1.0

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
    features['numTheirFood'] = numTheirFood
    features['distToTheirFoodCenter'] = distToTheirFoodCenter
    features['minDistToFood'] = minDistToFood
    features['minDistToCapsule'] = minDistToCapsule
    features['minDistToGhost'] = numTheirFood
    features['ourOff'] = numTheirFood
    features['theirDef'] = numTheirFood
    features['numOurFood'] = numOurFood
    features['numOurDef'] = numOurDef
    features['numTheirOff'] = numTheirOff
    features['minDistToInvader'] = minDistToInvader
    features['distToOurFoodBorderCenter'] = distToOurFoodBorderCenter
    features['minDistToHomie'] = minDistToHomie
    features['score'] = score
    features['stop'] = stop
    features['reverse'] = reverse
    features['bias'] = bias
    return features

class KeyboardAgent2(ReflexCaptureAgent):
  """
  A second agent controlled by the keyboard.
  """
  # NOTE: Arrow keys also work.
  WEST_KEY  = 'j'
  EAST_KEY  = "l"
  NORTH_KEY = 'i'
  SOUTH_KEY = 'k'
  STOP_KEY  = 'u'

  def __init__( self, index = 0 ):
    self.lastMove = Directions.STOP
    self.index = index
    self.keys = []
    self.befores = []
    self.afters = []

  def getAction(self, state):

    self.befores.append(self.getFeatures(state).values())

    from graphicsUtils import keys_waiting
    from graphicsUtils import keys_pressed
    keys = keys_waiting() + keys_pressed()
    if keys != []:
      self.keys = keys

    legal = state.getLegalActions(self.index)
    move = self.getMove(legal)

    if move == Directions.STOP:
      # Try to move in the same direction as before
      if self.lastMove in legal:
        move = self.lastMove

    if (self.STOP_KEY in self.keys) and Directions.STOP in legal: move = Directions.STOP

    if move not in legal:
      move = random.choice(legal)

    self.lastMove = move

    self.afters.append(self.getFeaturesAfterAction(state, move).values())
    global GAME_ID
    pickle.dump(self.befores, open('training/guneet-befores-' + GAME_ID + '.pkl', 'w'))
    pickle.dump(self.afters, open('training/guneet-afters-' + GAME_ID + '.pkl', 'w'))

    return move

  def getMove(self, legal):
    move = Directions.STOP
    if   (self.WEST_KEY in self.keys or 'Left' in self.keys) and Directions.WEST in legal:  move = Directions.WEST
    if   (self.EAST_KEY in self.keys or 'Right' in self.keys) and Directions.EAST in legal: move = Directions.EAST
    if   (self.NORTH_KEY in self.keys or 'Up' in self.keys) and Directions.NORTH in legal:   move = Directions.NORTH
    if   (self.SOUTH_KEY in self.keys or 'Down' in self.keys) and Directions.SOUTH in legal: move = Directions.SOUTH
    return move

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


  def getPosition(self, index, gameState):
    """
    Only call this on enemy indices.
    """
    if gameState.getAgentState(index).getPosition() is not None:
      return gameState.getAgentState(index).getPosition()
    else:
      return self.getBeliefDistribution(index).argMax()

  def getFeaturesAfterAction(self, gameState, action):
    return self.getFeatures(self.getSuccessor(gameState, action))

  def getFeatures(self, succ):
    myPos = succ.getAgentState(self.index).getPosition()
    width = succ.getWalls().width
    height = succ.getWalls().height
    xBorder = succ.getWalls().width / 2
    squad = [s for s in self.getTeam(succ) if s != self.index]
    ourFood = self.getFoodYouAreDefending(succ).asList()
    OUR = self.getCapsulesYouAreDefending(succ)
    ourOff = [oo for oo in self.getTeam(succ) if succ.getAgentState(oo).isPacman]
    ourDef = [od for od in self.getTeam(succ) if not succ.getAgentState(od).isPacman]
    theirFood = self.getFood(succ).asList()
    THEIR = self.getCapsules(succ)
    theirOff = [to for to in self.getOpponents(succ) if succ.getAgentState(to).isPacman]
    theirDef = [td for td in self.getOpponents(succ) if not succ.getAgentState(td).isPacman]

    self.updateSharedInfo(succ, myPos)

    ### features
    ## offense
    # len(theirFood)
    # distToTheirFoodCenter
    # minDistToFood
    # minDistToCapsule
    # minDistToGhost
    # scaredTime (TODO)
    # recentlyEaten (TODO)
    # ourOff
    # theirDef
    ## defense
    # len(ourFood)
    # ourDef
    # theirOff
    # minDistToInvader
    # distToOurFoodBorderCenter
    ## general
    # minDistToHomie
    # score
    # stop
    # reverse
    # bias
    # ??? combine ourOffRatio and theirOffRatio ???

    ## offense
    numTheirFood = len(theirFood)

    tfc = [0, 0]
    for tf in theirFood:
      tfc = list(sum(c) for c in zip(tfc, tf))
    tfc = list(c / len(theirFood) for c in tfc)
    if tfc not in DeepAgent.legalPositions:
      for lp in DeepAgent.legalPositions:
        if util.manhattanDistance(tfc, lp) == 2:
          tfc = lp
          break
    distToTheirFoodCenter = self.getMazeDistance(myPos, tuple(tfc))

    minDistToFood = width + height
    for tf in theirFood:
      minDistToFood = min(minDistToFood, self.getMazeDistance(myPos, tf))

    minDistToCapsule = 0 if len(THEIR) == 0 else (width + height)
    for tc in THEIR:
      minDistToCapsule = min(minDistToCapsule, self.getMazeDistance(myPos, tc))

    minDistToGhost = width + height
    for td in theirDef:
      minDistToGhost = min(minDistToGhost, self.getMazeDistance(myPos, self.getPosition(td, succ)))

    numOurOff = len(ourOff)
    numTheirDef = len(theirDef)

    ## defense
    numOurFood = len(ourFood)
    numTheirOff = len(theirOff)
    numOurDef = len(ourDef)

    minDistToInvader = 0 if len(theirOff) == 0 else (width + height)
    for to in theirOff:
      minDistToInvader = min(minDistToInvader, self.getMazeDistance(myPos, self.getPosition(to, succ)))

    ofc = [0, 0]
    for of in ourFood:
      ofc = list(sum(c) for c in zip(ofc, of))
    ofc = list(c / len(ourFood) for c in ofc)
    ofc[0] = (ofc[0] + xBorder) / 2
    if ofc not in DeepAgent.legalPositions:
      for lp in DeepAgent.legalPositions:
        if util.manhattanDistance(ofc, lp) == 1:
          ofc = lp
          break
    distToOurFoodBorderCenter = self.getMazeDistance(myPos, tuple(ofc))

    ## general
    minDistToHomie = width + height
    for s in squad:
      dist = self.getMazeDistance(myPos, succ.getAgentState(s).getPosition())
      minDistToHomie = min(dist, minDistToHomie)

    score = self.getScore(succ)
    stop = 1.0
    reverse = 1.0
    bias = 1.0

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
    features['numTheirFood'] = numTheirFood
    features['distToTheirFoodCenter'] = distToTheirFoodCenter
    features['minDistToFood'] = minDistToFood
    features['minDistToCapsule'] = minDistToCapsule
    features['minDistToGhost'] = numTheirFood
    features['ourOff'] = numTheirFood
    features['theirDef'] = numTheirFood
    features['numOurFood'] = numOurFood
    features['numOurDef'] = numOurDef
    features['numTheirOff'] = numTheirOff
    features['minDistToInvader'] = minDistToInvader
    features['distToOurFoodBorderCenter'] = distToOurFoodBorderCenter
    features['minDistToHomie'] = minDistToHomie
    features['score'] = score
    features['stop'] = stop
    features['reverse'] = reverse
    features['bias'] = bias
    return features
