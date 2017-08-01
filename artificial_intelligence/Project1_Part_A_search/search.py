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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def helper(com_state, closed_set, actions, problem):
    state, act, _ = com_state
    actions.append(act)
    if problem.isGoalState(state):
        return True
    if state not in closed_set:
        closed_set.append(state)

    for each in problem.getSuccessors(state):
        tmp_state, _, _ = each
        if tmp_state in closed_set:
            continue
        if helper(each, closed_set, actions, problem):
            return True;

    actions.pop()
    return False

def helper_1(com_state, closed_set, problem):
    state, actions = com_state
    if problem.isGoalState(state):
        return (True, actions[1:])
    if state not in closed_set:
        closed_set.append(state)
        for each in problem.getSuccessors(state):
            tmp_state, tmp_act, _ = each
            #if tmp_state in closed_set:
            #    continue
            ret = helper_1((tmp_state, actions+[tmp_act]), closed_set, problem)
            if (ret[0]):
                return ret
    return (False, [])


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
    "*** YOUR CODE HERE ***"
    """
    closed_set = []
    actions = []
    start_state = problem.getStartState()

    _ = helper((start_state, 0, 0), closed_set, actions, problem)
    return actions[1:]
    """
    closed_set = []
    start_state = problem.getStartState()

    _, actions = helper_1((start_state, [0]), closed_set, problem)
    return actions



    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    closed_set = []
    start_state = problem.getStartState()
    #print("start state: ", start_state)
    queue.push((start_state, [0]))
    while not queue.isEmpty():
        state, actions = queue.pop()
        if problem.isGoalState(state):
            #print(closed_set)
            #print(actions[1:])
            return actions[1:]
        if state not in closed_set:
            closed_set.append(state)
            for each in problem.getSuccessors(state):
                tmp_state, tmp_act, _ = each
                #if tmp_state in closed_set:
                #    continue
                queue.push((tmp_state, actions+[tmp_act]))


    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    queue = util.PriorityQueue()
    closed_set = []
    start_state = problem.getStartState()
    queue.push((start_state, [0], 0), 0)
    while not queue.isEmpty():
        state, actions, sum_cost = queue.pop()
        if problem.isGoalState(state):
            return actions[1:]
        if state not in closed_set:
            closed_set.append(state)
            for each in problem.getSuccessors(state):
                tmp_state, tmp_act, cost = each
                #if tmp_state in closed_set:
                #    continue
                queue.push((tmp_state, actions+[tmp_act], sum_cost+cost), sum_cost+cost)



    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    queue = util.PriorityQueue()
    closed_set = []
    start_state = problem.getStartState()
    queue.push((start_state, [0], 0), 0)
    while not queue.isEmpty():
        state, actions, sum_cost = queue.pop()
        if problem.isGoalState(state):
            return actions[1:]
        if state not in closed_set:
            closed_set.append(state)
            for each in problem.getSuccessors(state):
                tmp_state, tmp_act, cost = each
                #if tmp_state in closed_set:
                #    continue
                h = heuristic(tmp_state, problem)
                queue.push((tmp_state, actions+[tmp_act], sum_cost+cost), sum_cost+cost+h)
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
