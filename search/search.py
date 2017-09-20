# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

from game import Directions

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    stack = util.Stack()
    stack.push(problem.getStartState())
    back = {}
    expanded = []

    while not stack.isEmpty():
        current = stack.pop()

        if problem.isGoalState(current):
            actions = []
            while current != problem.getStartState():
                actions.append(back[current][1])
                current = back[current][0]
            actions.reverse()
            return actions

        expanded.append(current)
        for child in problem.getSuccessors(current):
            coords = child[0]
            if coords not in expanded:
                stack.push(coords)
                back[coords] = (current, child[1])

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    start = problem.getStartState()
    q = util.Queue()
    q.push(start)
    back = {}
    back[start] = None

    while not q.isEmpty():
        current = q.pop()

        if problem.isGoalState(current):
            actions = []
            while current != start:
                actions.append(back[current][1]) # get the action
                current = back[current][0]
            actions.reverse()
            return actions

        for child in problem.getSuccessors(current):
            state = child[0]
            if state not in back:
                q.push(state)
                back[state] = (current, child[1])

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    pq = util.PriorityQueue()
    pq.update(problem.getStartState(), 0)
    back = {}
    back[problem.getStartState()] = (None, None, 0)
    expanded = []

    if problem.isGoalState(problem.getStartState()):
        return []

    while not pq.isEmpty():
        current = pq.pop()
        currentCost = back[current][2]

        if problem.isGoalState(current):
            actions = []
            while current != problem.getStartState():
                actions.append(back[current][1])
                current = back[current][0]
            actions.reverse()
            return actions

        expanded.append(current)
        for child in problem.getSuccessors(current):
            coords = child[0]
            if coords not in expanded:
                cost = currentCost + child[2]
                if not (coords in back and cost > back[coords][2]):
                    pq.update(coords, cost)
                    back[coords] = (current, child[1], cost)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    start = problem.getStartState()
    pq = util.PriorityQueue()
    pq.push(start, heuristic(start, problem))
    back = {}
    back[start] = (None, None, 0) # back now includes cost
    expanded = []

    while not pq.isEmpty():
        current = pq.pop()
        currentCost = back[current][2]

        if problem.isGoalState(current):
            actions = []
            while current != problem.getStartState():
                actions.append(back[current][1])
                current = back[current][0]
            actions.reverse()
            return actions

        expanded.append(current)
        for child in problem.getSuccessors(current):
            state = child[0]
            if state not in expanded:
                childCost = currentCost + child[2]
                if not (state in back and childCost > back[state][2]):
                    pq.update(state, childCost + heuristic(state, problem))
                    back[state] = (current, child[1], childCost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
