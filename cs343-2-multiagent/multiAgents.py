# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        eval = 0

        # only care about nearby ghosts
        ghostThreat = manhattanDistance(newPos, newGhostStates[0].getPosition())
        if ghostThreat < 5:
            eval -= (5 - ghostThreat) ** 5

        # incentivize eating food
        maxFood = newFood.height * newFood.width
        eval += (maxFood - newFood.count()) ** 2

        # get closer to food
        minFoodDist = newFood.height + newFood.width
        for x in range(newFood.width):
            for y in range(newFood.height):
                if newFood[x][y]:
                    foodDist = manhattanDistance(newPos, (x, y))
                    minFoodDist = min(minFoodDist, foodDist)
        eval -= minFoodDist

        return eval

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        maxi = -float('inf')
        best = None
        actions = gameState.getLegalActions(0)
        # seperate the root node from the recursion because we need to return the best action
        for a in actions:
            temp = self.recurse(1, gameState.generateSuccessor(0, a))
            if temp > maxi:
                maxi = temp
                best = a
        return best

    def recurse(self, level, state):
        index = level % state.getNumAgents()
        # if depth is reached or legal actions available, evaluate the state
        if level == self.depth * state.getNumAgents() or len(state.getLegalActions(index)) == 0:
            return self.evaluationFunction(state)
        if index == 0: # pacman (max)
            actions = state.getLegalActions(index)
            maxi = -float('inf')
            for a in actions:
                maxi = max(maxi, self.recurse(level + 1, state.generateSuccessor(index, a)))
            return maxi
        else: # ghost (min)
            actions = state.getLegalActions(index)
            mini = float('inf')
            for a in actions:
                mini = min(mini, self.recurse(level + 1, state.generateSuccessor(index, a)))
            return mini

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = -float('inf')
        beta = float('inf')

        maxi = -float('inf')
        best = None
        actions = gameState.getLegalActions(0)
        # same algorithm as minimax except we pass alpha and beta
        for a in actions:
            temp = self.recurse(1, gameState.generateSuccessor(0, a), alpha, beta)
            if temp > maxi:
                maxi = temp
                best = a
            if maxi > beta:
                break
            alpha = max(alpha, maxi)
        return best

    def recurse(self, level, state, alpha, beta):
        index = level % state.getNumAgents()
        if level == self.depth * state.getNumAgents() or len(state.getLegalActions(index)) == 0:
            return self.evaluationFunction(state)
        if index == 0: # pacman (max)
            actions = state.getLegalActions(index)
            maxi = -float('inf')
            for a in actions:
                maxi = max(maxi, self.recurse(level + 1, state.generateSuccessor(index, a), alpha, beta))
                if maxi > beta: # if we find eval greater than beta, then prune rest of the branches
                    return maxi
                alpha = max(alpha, maxi)
            return maxi
        else: # ghost (min)
            actions = state.getLegalActions(index)
            mini = float('inf')
            for a in actions:
                mini = min(mini, self.recurse(level + 1, state.generateSuccessor(index, a), alpha, beta))
                if mini < alpha: # if we find eval less than alpha, then prune rest of branches
                    return mini
                beta = min(beta, mini)
            return mini

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        maxi = -float('inf')
        best = None
        actions = gameState.getLegalActions(0)
        for a in actions:
            temp = self.recurse(1, gameState.generateSuccessor(0, a))
            if temp > maxi:
                maxi = temp
                best = a
        return best

    def recurse(self, level, state):
        index = level % state.getNumAgents()
        if level == self.depth * state.getNumAgents() or len(state.getLegalActions(index)) == 0:
            return self.evaluationFunction(state)
        if index == 0: # pacman
            actions = state.getLegalActions(index)
            maxi = -float('inf')
            for a in actions:
                maxi = max(maxi, self.recurse(level + 1, state.generateSuccessor(index, a)))
            return maxi
        else: # ghost
            actions = state.getLegalActions(index)
            mini = 0
            for a in actions:
                mini += self.recurse(level + 1, state.generateSuccessor(index, a))
            return mini/len(actions) # instead of returning the minimum, return the average eval


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    # EVALUATION IDEAS:
    # incentivize power pellets when ghosts are nearby
    # stay a safe distance away from ghosts
    # eat food
    # get close to food
    # get close to ghosts when scared in consideration of scared time left

    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    eval = 0

    # only care about nearby ghosts
    ghostThreat = manhattanDistance(newPos, newGhostStates[0].getPosition())
    if ghostThreat < 5:
        eval -= (5 - ghostThreat) ** 5

    # want to get loser to ghost if it is scared
    scaredTime = newScaredTimes[0]
    if scaredTime > 0: # if the ghost is scared, ignore the ghostThreat
        eval = 0
    if ghostThreat < scaredTime: # if pacman can reach the ghost within the scaredTime, go eat him
        eval += (scaredTime - ghostThreat) ** 3

    # incentivize eating food
    maxFood = newFood.height * newFood.width
    eval += (maxFood - newFood.count()) ** 2

    # get closer to food
    minFoodDist = newFood.height + newFood.width
    for x in range(newFood.width):
        for y in range(newFood.height):
            if newFood[x][y]:
                foodDist = manhattanDistance(newPos, (x, y))
                minFoodDist = min(minFoodDist, foodDist)
    eval -= minFoodDist

    return eval

# Abbreviation
better = betterEvaluationFunction
