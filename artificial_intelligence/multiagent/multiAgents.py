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


        return successorGameState.getScore()

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
      Your minimax agent (question 1)
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

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # there is no need to check isWin or isLose
        scores = []
        legalMoves = gameState.getLegalActions(0)
        for move in legalMoves:
            next_state = gameState.generateSuccessor(0, move)
            score = self.minimax(1, 1, next_state)
            scores.append(score)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]


        #util.raiseNotDefined()
    def minimax(self, cur_dep, idx, state):
        # if Win or Lose, there is no need to expand again
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        num_agents = state.getNumAgents()
        legalMoves = state.getLegalActions(idx)

        scores = []
        if cur_dep==self.depth and idx==num_agents-1:
            for move in legalMoves:
                next_state = state.generateSuccessor(idx, move)
                score = self.evaluationFunction(next_state)
                scores.append(score)
            return min(scores)
        for move in legalMoves:
            next_state = state.generateSuccessor(idx, move)
            if idx != num_agents-1:
                score = self.minimax(cur_dep, idx+1, next_state)
            else:
                score = self.minimax(cur_dep+1, 0, next_state)
            scores.append(score)
        if idx == 0:
            return max(scores)
        else:
            return min(scores)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        scores = []
        legalMoves = gameState.getLegalActions(0)
        alpha = -9999999
        beta = 9999999
        tmp_max = -9999999
        for move in legalMoves:
            next_state = gameState.generateSuccessor(0, move)
            score = self.alphaBeta(1, 1, next_state, alpha, beta)
            if score > tmp_max:
                tmp_max = score
                bestMove = move

            if tmp_max > alpha:
                alpha = tmp_max

        return bestMove

        #util.raiseNotDefined()

    def alphaBeta(self, cur_dep, idx, state, alpha, beta):
        if state.isWin() or state.isLose(): # then generateSuccessor will return empty list
            return self.evaluationFunction(state)

        num_agents = state.getNumAgents()
        legalMoves = state.getLegalActions(idx)

        # assuming there must have ghosts
        scores = []

        tmp_min = 9999999
        tmp_max = -9999999

        # max depth
        if cur_dep==self.depth and idx==num_agents-1:
            for move in legalMoves:
                next_state = state.generateSuccessor(idx, move)
                score = self.evaluationFunction(next_state)
                if score < tmp_min:
                    tmp_min = score;
                """
                if idx-1 == 0: # previous agent is pacman
                    if tmp_min < alpha:
                        return tmp_min
                """
                if tmp_min < alpha:
                    return tmp_min

                if tmp_min < beta:
                    beta = tmp_min
            return tmp_min

        # max node
        if idx == 0:
            for move in legalMoves:
                next_state = state.generateSuccessor(idx, move)
                score = self.alphaBeta(cur_dep, 1, next_state, alpha, beta)
                if score > tmp_max:
                    tmp_max = score

                if tmp_max > beta:
                    return tmp_max
                if tmp_max > alpha:
                    alpha = tmp_max
            return tmp_max

        # min node
        for move in legalMoves:
            next_state = state.generateSuccessor(idx, move)
            if idx == num_agents-1:
                score = self.alphaBeta(cur_dep+1, 0, next_state, alpha, beta)
            else:
                score = self.alphaBeta(cur_dep, idx+1, next_state, alpha, beta)
            if score < tmp_min:
                tmp_min = score
            """
            if idx-1 == 0:
                if tmp_min < alpha:
                    return tmp_min
            """
            if tmp_min < alpha:
                return tmp_min

            if tmp_min < beta:
                beta = tmp_min
        return tmp_min

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        scores = []
        legalMoves = gameState.getLegalActions(0)
        for move in legalMoves:
            next_state = gameState.generateSuccessor(0, move)
            score = self.expectimax(1, 1, next_state)
            scores.append(score)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]


        #util.raiseNotDefined()
    def expectimax(self, cur_dep, idx, state):
        # if Win or Lose, there is no need to expand again
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        num_agents = state.getNumAgents()
        legalMoves = state.getLegalActions(idx)

        scores = []
        if cur_dep==self.depth and idx==num_agents-1:
            for move in legalMoves:
                next_state = state.generateSuccessor(idx, move)
                score = self.evaluationFunction(next_state)
                scores.append(score)
            tmp_sum = 0.0
            for i in xrange(len(scores)):
                tmp_sum += scores[i]
            return tmp_sum/len(scores)
        for move in legalMoves:
            next_state = state.generateSuccessor(idx, move)
            if idx != num_agents-1:
                score = self.expectimax(cur_dep, idx+1, next_state)
            else:
                score = self.expectimax(cur_dep+1, 0, next_state)
            scores.append(score)
        if idx == 0:
            return max(scores)
        else:
            tmp_sum = 0.0
            for i in xrange(len(scores)):
                tmp_sum += scores[i]
            return tmp_sum/len(scores)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 4).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    pac_pos = currentGameState.getPacmanPosition()
    """
    capsules = currentGameState.getCapsules()
    cap_dis = 0.0
    if capsules:
        min_dis = 9999999
        for i in xrange(len(capsules)):
            man_dis = manhattanDistance(pac_pos, capsules[i])
            if man_dis < min_dis:
                min_dis = man_dis
        cap_dis = 0.5/min_dis

    """
    food_grid = currentGameState.getFood()
    width = food_grid.width
    height = food_grid.height
    min_dis = 9999999
    for x in xrange(width):
        for y in xrange(height):
            if food_grid[x][y]:
                tmp = manhattanDistance(pac_pos, (x, y))
                if tmp < min_dis:
                    min_dis = tmp;

    food_dis = 10.0/(min_dis+1)
    return currentGameState.getScore()/(currentGameState.getNumFood() + 1) + food_dis
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

