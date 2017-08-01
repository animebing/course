# factorOperations.py
# -------------------
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


from bayesNet import Factor
import operator as op
import util

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors, joinVariable):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that
        contain that variable.

        Returns a tuple of
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print "Factor failed joinFactorsByVariable typecheck: ", factor
            raise ValueError, ("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))

        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()


def joinFactors(factors):
    """
    Question 3: Your join implementation

    Input factors is a list of factors.

    You should calculate the set of unconditioned variables and conditioned
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input
    (such as getProbability and setProbability) can handle
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print "Factor failed joinFactors typecheck: ", factor
            raise ValueError, ("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) +
                    "\nappear in more than one input factor.\n" +
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))

    new_factor = factors[0]
    var_domain_dict = factors[0].variableDomainsDict()

    for i in xrange(1, len(factors)):
        uncond_pre = new_factor.unconditionedVariables()
        cond_pre = new_factor.conditionedVariables()
        var_pre = uncond_pre | cond_pre

        uncond_cur = factors[i].unconditionedVariables()
        cond_cur = factors[i].conditionedVariables()
        var_cur = uncond_cur | cond_cur

        union_uncond = uncond_pre | uncond_cur
        union_cond = cond_pre | cond_cur
        new_uncond = union_uncond
        new_cond = union_cond-union_uncond

        tmp_factor = Factor(new_uncond, new_cond, var_domain_dict)
        all_assigns = tmp_factor.getAllPossibleAssignmentDicts()
        for assign in all_assigns:
            assign_pre = {}
            assign_cur = {}
            for var in assign:
                if var in var_pre:
                    assign_pre[var] = assign[var]
                if var in var_cur:
                    assign_cur[var] = assign[var]

            prob_pre = new_factor.getProbability(assign_pre)
            prob_cur = factors[i].getProbability(assign_cur)
            prob_tmp = prob_pre*prob_cur
            tmp_factor.setProbability(assign, prob_tmp)
        new_factor = tmp_factor

    return new_factor

    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()


def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor, eliminationVariable):
        """
        Question 4: Your eliminate implementation

        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.

        You should calculate the set of unconditioned variables and conditioned
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print "Factor failed eliminate typecheck: ", factor
            raise ValueError, ("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" +
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))

        if len(factor.unconditionedVariables()) == 1:
            print "Factor failed eliminate typecheck: ", factor
            raise ValueError, ("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))


        var_domain_dict = factor.variableDomainsDict()
        eli_vals = var_domain_dict[eliminationVariable]
        var_uncond = factor.unconditionedVariables()
        var_cond = factor.conditionedVariables()

        new_uncond = set(var_uncond)
        new_uncond.remove(eliminationVariable)
        new_factor = Factor(new_uncond, var_cond, var_domain_dict)
        all_assigns = new_factor.getAllPossibleAssignmentDicts()
        for assign in all_assigns:
            prob = 0.0
            tmp_assign = dict(assign)
            for val in eli_vals:
                tmp_assign[eliminationVariable] = val;
                prob += factor.getProbability(tmp_assign)

            new_factor.setProbability(assign, prob)

        return new_factor
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

    return eliminate

eliminate = eliminateWithCallTracking()


def normalize(factor):
    """
    Question 5: Your normalize implementation

    Input factor is a single factor.

    The set of conditioned variables for the normalized factor consists
    of the input factor's conditioned variables as well as any of the
    input factor's unconditioned variables with exactly one entry in their
    domain.  Since there is only one entry in that variable's domain, we
    can either assume it was assigned as evidence to have only one variable
    in its domain, or it only had one entry in its domain to begin with.
    This blurs the distinction between evidence assignments and variables
    with single value domains, but that is alright since we have to assign
    variables that only have one value in their domain to that single value.

    Return a new factor where the sum of the all the probabilities in the table is 1.
    This should be a new factor, not a modification of this factor in place.

    If the sum of probabilities in the input factor is 0,
    you should return None.

    This is intended to be used at the end of a probabilistic inference query.
    Because of this, all variables that have more than one element in their
    domain are assumed to be unconditioned.
    There are more general implementations of normalize, but we will only
    implement this version.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """
    # typecheck portion
    variableDomainsDict = factor.variableDomainsDict()
    #print(variableDomainsDict)
    for conditionedVariable in factor.conditionedVariables():
        if len(variableDomainsDict[conditionedVariable]) > 1:
            print "Factor failed normalize typecheck: ", factor
            raise ValueError, ("The factor to be normalized must have only one " + \
                            "assignment of the \n" + "conditional variables, " + \
                            "so that total probability will sum to 1\n" +
                            str(factor))



    var_uncond = factor.unconditionedVariables()
    var_cond = factor.conditionedVariables()
    if len(var_uncond) == 1:
        total_prob = 0.0
        all_assigns = factor.getAllPossibleAssignmentDicts()
        for assign in all_assigns:
            total_prob += factor.getProbability(assign)

        new_factor = Factor(var_uncond, var_cond, variableDomainsDict)
        for assign in all_assigns:
            prob = factor.getProbability(assign)
            prob /= total_prob
            new_factor.setProbability(assign, prob)

        return new_factor

    s_uncond = set()
    m_uncond = set()
    for var in var_uncond:
        if len(variableDomainsDict[var]) > 1:
            m_uncond.add(var)
        else:
            s_uncond.add(var)

    new_uncond = m_uncond
    new_cond = s_uncond | var_cond
    new_factor = Factor(new_uncond, new_cond, variableDomainsDict)

    tmp_factor = factor
    for var in m_uncond:
        tmp_factor = eliminate(tmp_factor, var)
    all_assigns = tmp_factor.getAllPossibleAssignmentDicts()
    tmp_prob = tmp_factor.getProbability(all_assigns[0])
    all_assigns = new_factor.getAllPossibleAssignmentDicts()
    for assign  in all_assigns:
        prob = factor.getProbability(assign)
        prob /= tmp_prob
        new_factor.setProbability(assign, prob)


    return new_factor
    "*** YOUR CODE HERE ***"
    #return None
    util.raiseNotDefined()

