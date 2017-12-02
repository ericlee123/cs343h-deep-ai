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
import math

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

        action = random.choice(bestActions)
        return action

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
    inferences = {}

    def initialize(self, gameState):
        if DeepAgent.legalPositions is None:
          DeepAgent.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]

    def doInference(self, gameState, myPos):
        if not DeepAgent.init:
            self.initialize(gameState)

        for opp in self.getOpponents(gameState):
            if opp not in DeepAgent.inferences:
                DeepAgent.inferences[opp] = self.getUniformDistribution()

            # do the inference
            emissionModel = util.Counter()
            # hardcoded uniform distribution over [-6, +6]
            observation = gameState.getAgentDistances()[opp]
            for i in range(observation-6, observation+7):
                emissionModel[i] = 1.0 / 13

            allPossible = util.Counter()
            for lp in DeepAgent.legalPositions:
                dist = util.manhattanDistance(lp, myPos)
                allPossible[lp] = emissionModel[dist] * DeepAgent.inferences[opp][lp]
            allPossible.normalize()

            if allPossible.totalCount() != 0:
                DeepAgent.inferences[opp] = allPossible
            else:
                DeepAgent.inferences[opp] = self.getUniformDistribution()

    def getUniformDistribution(self):
        uniform = util.Counter()
        for lp in DeepAgent.legalPositions:
            uniform[lp] = 1.0 / len(DeepAgent.legalPositions)
        return uniform

    def getPosition(self, index, gameState):
        """
        Only call this on enemy indices.
        """
        if gameState.getAgentState(index).getPosition() is not None:
            return gameState.getAgentState(index).getPosition()
        else:
            return DeepAgent.inferences[index].argMax()

    def getFeatures(self, gameState, action):
        succ = self.getSuccessor(gameState, action)

        myPos = succ.getAgentState(self.index).getPosition()
        width = succ.getWalls().width
        height = succ.getWalls().height
        xBorder = succ.getWalls().width / 2
        squad = [s for s in self.getTeam(succ) if s != self.index]
        ourFood = self.getFoodYouAreDefending(succ).asList()
        OUR = self.getCapsulesYouAreDefending(succ)
        ourOff = [oo for oo in self.getTeam(succ) if succ.getAgentState(oo).isPacman]
        ourDef = [od for od in self.getTeam(succ) if not succ.getAgentState(od).isPacman]
        theirFood = self.getFood(gameState).asList()
        THEIR = self.getCapsules(succ)
        theirOff = [to for to in self.getOpponents(succ) if succ.getAgentState(to).isPacman]
        theirDef = [td for td in self.getOpponents(succ) if not succ.getAgentState(td).isPacman]

        self.doInference(gameState, myPos)

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
        minDist = width + height
        moreClosest = tfc
        if moreClosest not in DeepAgent.legalPositions:
            for lp in DeepAgent.legalPositions:
                dist = util.manhattanDistance(tuple(tfc), lp)
                if dist < minDist:
                    minDist = dist
                    moreClosest = lp
        distToTheirFoodCenter = self.getMazeDistance(myPos, tuple(moreClosest))

        minDistToFood = width + height
        for tf in theirFood:
          minDistToFood = min(minDistToFood, self.getMazeDistance(myPos, tf))
        # print len(theirFood)

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
        minDist = width + height
        closest = ofc
        if closest not in DeepAgent.legalPositions:
            for lp in DeepAgent.legalPositions:
                dist = util.manhattanDistance(tuple(ofc), lp)
                if dist < minDist:
                    minDist = dist
                    closest = lp
        distToOurFoodBorderCenter = self.getMazeDistance(myPos, tuple(closest))

        ## general
        minDistToHomie = width + height
        for s in squad:
          dist = self.getMazeDistance(myPos, succ.getAgentState(s).getPosition())
          minDistToHomie = min(dist, minDistToHomie)

        # score = self.getScore(succ)
        # stop = 1.0
        # reverse = 1.0
        # bias = 1.0
        #
        # # # first order features
        # # ourFood = self.getFood(gameState).asList()
        # # theirFood = self.getFoodYouAreDefending(gameState).asList()
        # # OUR = self.getCapsules(gameState)
        # # THEIR = self.getCapsulesYouAreDefending(gameState)
        # # squad = [s for s in self.getTeam(gameState) if s != self.index]
        # # them = self.getOpponents(gameState)
        # # score = self.getScore(gameState)
        # # succ = self.getSuccessor(gameState, action)
        # # pos = succ.getAgentState(self.index).getPosition()
        # # ourOff = len([oo for oo in self.getTeam(succ) if succ.getAgentState(oo).isPacman])
        # # ourDef = len(self.getTeam(succ)) - ourOff
        # # theirOff = len([to for to in self.getOpponents(succ) if succ.getAgentState(to).isPacman])
        # # theirDef = len(self.getTeam(succ)) - theirOff
        # # # our/their recently eaten (need persistent state)
        # # squadPos = [succ.getAgentState(sp).getPosition() for sp in self.getTeam(gameState)]
        # # # theirPos (need bayesian inference or particle filtering)
        # #
        # # # second order features
        # # distToHomie = self.getMazeDistance(pos, succ.getAgentState(squad[0]).getPosition())
        # # ourOffRatio = len(theirFood) / (ourOff + 1)
        # # theirOffRatio = len(ourFood) / (theirOff + 1)

        features = util.Counter()
        # offense (scale by offense defense ratio)
        features['numTheirFood'] = numTheirFood
        features['distToTheirFoodCenter'] = distToTheirFoodCenter
        features['minDistToFood'] = minDistToFood
        features['minDistToGhost'] = minDistToGhost
        features['numOurOff'] = numOurOff
        features['numTheirDef'] = numTheirDef
        # defense
        features['numOurFood'] = numOurFood
        features['numTheirOff'] = numTheirOff
        features['numOurDef'] = numOurDef
        features['minDistToInvader'] = minDistToInvader
        features['distToOurFoodBorderCenter'] = distToOurFoodBorderCenter
        features['minDistToHomie'] = math.sqrt(minDistToHomie)
        return features

    def getWeights(self, gameState, action):
        return {
        # offense
            # 'numTheirFood'              : -20,
            # 'distToTheirFoodCenter'     : -5,
            # 'minDistToFood'             : -100,
            # 'minDistToGhost'            : 10,
            # 'numOurOff'                 : 10,
            # 'numTheirDef'               : -10,
        # defense
            'numOurFood'                : 10,
            'numTheirOff'               : -10,
            'numOurDef'                 : 10,
            'minDistToInvader'          : -100,
            'distToOurFoodBorderCenter' : -10,
            'minDistToHomie'            : 60
        }
