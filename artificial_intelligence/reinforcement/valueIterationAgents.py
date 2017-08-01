# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        all_states = self.mdp.getStates()

        for i in xrange(self.iterations):
            value_cur = util.Counter()
            for each_state in all_states:
                actions = self.mdp.getPossibleActions(each_state)
                if len(actions) == 0:
                    value_cur[each_state] = 0
                    continue

                if self.mdp.isTerminal(each_state):
                    trans_model = self.mdp.getTransitionStatesAndProbs(each_state, actions[0])
                    exit_reward = self.mdp.getReward(each_state, action[0], trans_model[0][0])
                    value_cur[each_state] = exit_reward
                    continue
                q_values = []
                for each_action in actions:
                    q_values.append(self.computeQValueFromValues(each_state, each_action))
                value_cur[each_state] = max(q_values)
            self.values = value_cur.copy()



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        trans_model = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0.0
        for i in xrange(len(trans_model)):
            next_state, prob = trans_model[i]
            reward = self.mdp.getReward(state, action, next_state)
            q_value += prob * (reward + self.discount * self.values[next_state])

        return q_value
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        all_actions = self.mdp.getPossibleActions(state)
        q_values = util.Counter()
        for each_action in all_actions:
            q_values[each_action] = self.computeQValueFromValues(state, each_action)

        return q_values.argMax()
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        all_states = self.mdp.getStates()

        for i in xrange(self.iterations):
            update_state = all_states[i%len(all_states)]
            actions = self.mdp.getPossibleActions(update_state)
            if len(actions) == 0:
                self.values[update_state] = 0
                continue

            if self.mdp.isTerminal(update_state):
                continue
            q_values = []
            for each_action in actions:
                q_values.append(self.computeQValueFromValues(update_state, each_action))
            self.values[update_state] = max(q_values)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        all_states = self.mdp.getStates()
        all_preds = {}
        for each_state in all_states:
            all_actions = self.mdp.getPossibleActions(each_state)
            if len(all_actions) == 0:
                continue
            # terminal state ?
            for each_action in all_actions:
                trans_model = self.mdp.getTransitionStatesAndProbs(each_state, each_action)
                for next_state, prob in trans_model:
                    if prob != 0.0:
                        if next_state not in all_preds.keys():
                            all_preds[next_state] = set()
                        all_preds[next_state].add(each_state)

        prior_queue = util.PriorityQueue()

        for each_state in all_states:
            all_actions = self.mdp.getPossibleActions(each_state)
            if len(all_actions) == 0:
                continue
            if self.mdp.isTerminal(each_state):
                continue

            cur_value = self.values[each_state]
            q_values = []
            for each_action in all_actions:
                q_values.append(self.computeQValueFromValues(each_state, each_action))
            should_value = max(q_values)
            prior_queue.update(each_state, -abs(cur_value-should_value))


        for i in xrange(self.iterations):

            if prior_queue.isEmpty():
                break

            tmp_state = prior_queue.pop()
            all_actions = self.mdp.getPossibleActions(tmp_state)
            q_values = []
            for each_action in all_actions:
                q_values.append(self.computeQValueFromValues(tmp_state, each_action))
            self.values[tmp_state] = max(q_values)

            for each_pred in all_preds[tmp_state]:
                all_actions = self.mdp.getPossibleActions(each_pred)
                cur_value = self.values[each_pred]
                q_values = []
                for each_action in all_actions:
                    q_values.append(self.computeQValueFromValues(each_pred, each_action))
                should_value = max(q_values)
                prior_queue.update(each_pred, -abs(cur_value-should_value))

