# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game


pacman_str = 'P'
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()

    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()

def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def sentence1():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.

    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** YOUR CODE HERE ***"
    A, B, C = logic.Expr('A'), logic.Expr('B'), logic.Expr('C')
    NOT_A, NOT_B = ~A, ~B
    E1 = A | B
    E2 = (NOT_A) % ((NOT_B) | C)
    E3 = logic.disjoin([NOT_A, NOT_B, C])
    return logic.conjoin([E1, E2, E3])

def sentence2():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.

    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** YOUR CODE HERE ***"
    A, B, C, D = logic.Expr('A'), logic.Expr('B'), logic.Expr('C'), logic.Expr('D')
    NOT_A, NOT_B, NOT_C, NOT_D = ~A, ~B, ~C, ~D
    E1 = (C) % (B | D)
    E2 = A >> (NOT_B & NOT_D)
    E3 = (~(B & NOT_C)) >> A
    E4 = NOT_D >> C
    return logic.conjoin([E1, E2, E3, E4])

def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the logic.PropSymbolExpr constructor, return a logic.PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive at time 0 and it was born at time 0.

    The Wumpus cannot both be alive at time 0 and be born at time 0.

    The Wumpus is born at time 0.
    """
    "*** YOUR CODE HERE ***"
    WumpusAlive_0 = logic.PropSymbolExpr('WumpusAlive', 0)
    WumpusAlive_1 = logic.PropSymbolExpr('WumpusAlive', 1)
    WumpusBorn_0 = logic.PropSymbolExpr('WumpusBorn', 0)
    WumpusKilled_0 = logic.PropSymbolExpr('WumpusKilled', 0)
    NOT_WumpusKilled_0, NOT_WumpusAlive_0 = ~WumpusKilled_0, ~WumpusAlive_0
    E1 = WumpusAlive_1 % ((WumpusAlive_0 & NOT_WumpusKilled_0) | (NOT_WumpusAlive_0 & WumpusBorn_0))
    E2 = ~(WumpusAlive_0 & WumpusBorn_0)
    E3 = WumpusBorn_0
    return logic.conjoin([E1, E2, E3])

def findModel(sentence):
    """Given a propositional logic sentence (i.e. a logic.Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** YOUR CODE HERE ***"
    return logic.pycoSAT(logic.to_cnf(sentence))

def atLeastOne(literals) :
    """
    Given a list of logic.Expr literals (i.e. in the form A or ~A), return a single
    logic.Expr instance in CNF (conjunctive normal form) that represents the logic
    that at least one of the literals in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    "*** YOUR CODE HERE ***"
    return logic.disjoin(literals)


def atMostOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in
    CNF (conjunctive normal form) that represents the logic that at most one of
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    l = []
    for e in literals:
        for e2 in literals:
            if e == e2:
                continue
            l.append(logic.disjoin([~e, ~e2]))
    return logic.conjoin(l)


def exactlyOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in
    CNF (conjunctive normal form)that represents the logic that exactly one of
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    return logic.conjoin([atMostOne(literals), atLeastOne(literals)])


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print plan
    ['West', 'South', 'North']
    """
    "*** YOUR CODE HERE ***"
    ret_actions = dict()
    for k, v in model.items():
        d, t = logic.PropSymbolExpr.parseExpr(k)
        if (d in actions ) and v:
            ret_actions[int(t)] = d
    return ret_actions.values()


def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    """
    "*** YOUR CODE HERE ***"
    Current_epxr = logic.PropSymbolExpr(pacman_str, x, y, t)
    action_delata = {
        'North': (0, 1),
        'South': (0, -1),
        'East': (1, 0),
        'West': (-1, 0)
    }
    Right_expr = []
    for action, d in action_delata.items():
        Action_epxr = logic.PropSymbolExpr(action, t-1)
        xx, yy = x - d[0], y - d[1]
        Position_expr = logic.PropSymbolExpr(pacman_str, xx, yy, t-1)
        if not walls_grid[xx][yy]:
            Right_expr.append(logic.conjoin([Action_epxr, Position_expr]))
    return Current_epxr % atLeastOne(Right_expr)

def getPositionSentence(startState, goalState, action_delata, walls):
    s = util.Queue()
    visited = set()
    StartExpr = logic.PropSymbolExpr(pacman_str, startState[0], startState[1], 0)
    s.push((startState, 0, StartExpr))
    while not s.isEmpty():
        curState, t, sentence = s.pop()
        if (curState == goalState):
            return sentence
        if curState in visited:
            continue
        visited.add(curState)
        for action, v in action_delata.items():
            nextState = (curState[0]+v[0], curState[1]+v[1])
            Action_expr = logic.PropSymbolExpr(action, t)
            #Position_expr = logic.PropSymbolExpr(pacman_str, nextState[0], nextState[1], t+1)
            #nextSentence = logic.conjoin([sentence, Action_expr, Position_expr])
            nextSentence = logic.conjoin([sentence, Action_expr])
            if not walls[nextState[0]][nextState[1]]:
                s.push((nextState, t+1, nextSentence))

def getFoodSentence(startState, action_delata, walls):
    s = util.Queue()
    visited = set()
    startPosition, startFoodGrid = startState
    StartExpr = logic.PropSymbolExpr(pacman_str, startPosition[0], startPosition[1], 0)
    s.push((startState, 0, StartExpr))
    while not s.isEmpty():
        curState, t, sentence = s.pop()
        curPosition, curFoodGrid = curState
        if (curFoodGrid.count() == 0):
            return sentence
        if curState in visited:
            continue
        visited.add(curState)
        for action, v in action_delata.items():
            nextPosition = (curPosition[0]+v[0], curPosition[1]+v[1])
            nextFoodGrid = curFoodGrid.copy()
            if nextFoodGrid[nextPosition[0]][nextPosition[1]]:
                nextFoodGrid[nextPosition[0]][nextPosition[1]] = False
            nextState = (nextPosition, nextFoodGrid)
            Action_expr = logic.PropSymbolExpr(action, t)
            #Position_expr = logic.PropSymbolExpr(pacman_str, nextPosition[0], nextPosition[1], t+1)
            #nextSentence = logic.conjoin([sentence, Action_expr, Position_expr])
            nextSentence = logic.conjoin([sentence, Action_expr])
            if not walls[nextPosition[0]][nextPosition[1]]:
                s.push((nextState, t+1, nextSentence))

def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    "*** YOUR CODE HERE ***"
    startState = problem.getStartState()
    goalState = problem.getGoalState()
    action_delata = {
        'North': (0, 1),
        'South': (0, -1),
        'East': (1, 0),
        'West': (-1, 0)
    }
    valid_actions = action_delata.keys()
    sentence = getPositionSentence(startState, goalState, action_delata, walls)
    model = findModel(sentence)
    print(model)
    return extractActionSequence(model, valid_actions)


def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    "*** YOUR CODE HERE ***"
    startState = problem.getStartState()
    action_delata = {
        'North': (0, 1),
        'South': (0, -1),
        'East': (1, 0),
        'West': (-1, 0)
    }
    valid_actions = action_delata.keys()
    sentence = getFoodSentence(startState, action_delata, walls)
    model = findModel(sentence)
    print(model)
    return extractActionSequence(model, valid_actions)
    util.raiseNotDefined()


# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)

