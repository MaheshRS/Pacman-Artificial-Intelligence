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
from game import Actions
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
        # Print statements for debugging.
        # print "=========================="
        # print "Legal Moves:", legalMoves
        # print "Scores:", scores
        # print "Best Score:", bestScore
        # print "Best Indices:", bestIndices;
        # print "=========================="

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

        "*** YOUR CODE HERE ***"
        # Get the path to the nearest food.
        food_distance = float("inf")
        for food_pos in successorGameState.getFood().asList():
            distance = util.manhattanDistance(newPos, food_pos)
            food_distance = min(food_distance, distance)

        # If the next direction is STOP then return zero.
        if action == Directions.STOP:
            return 0

        # Check for minimum ghost position.
        ghost_distance = float("inf")
        ghost_positions = successorGameState.getGhostPositions();
        for position in ghost_positions:
            distance_g = util.manhattanDistance(newPos, position);
            ghost_distance = min(distance_g, ghost_distance)

        # Calculating the scared time. If the ghost is scared then we need to approach the scared ghost.
        scared_time = 0
        for time in newScaredTimes:
            if time == 0:
                scared_time = 0
                break

            scared_time += time

        # It takes 2 positions for the ghost to eat Pac-man, then set the value to a large negative value.
        # So the Pac-Man moves away from the ghost.
        if ghost_distance < 3 and scared_time == 0:
            ghost_distance = -1000
        else:
            ghost_distance = 0

        # Set the cost of eating the nearest food.
        # The more near is the food, increase the evaluation value.
        food_distance = 1.0/food_distance

        # Make the Pac-man eat food, if there are less food than the previous state, motivate the Pac-man to eat more of them.
        food_count = len(newFood.asList())
        if len(newFood.asList()) < len(currentGameState.getFood().asList()):
            food_count = 1000

        # Return the linear evaluation value. Rounding the value to two points.
        return round(food_distance + ghost_distance + food_count, 2)


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
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # Start checking for the optimal move.
        value, action = self.recurrence(gameState, 0)
        return action

    # Check the terminal state (Is the state a win, lose or max depth i.e the end nodes are reached).
    def is_terminal_state(self, state, depth):
        if state.isWin() or state.isLose() or (depth == self.depth * state.getNumAgents()):
            return self.evaluationFunction(state), None
        else:
            return None, None

    # Using the below method to call the appropriate (max, min) functions with the correct agent index.
    def recurrence(self, state, depth):
        terminal_test, action = self.is_terminal_state(state, depth)
        if terminal_test is not None:
            return terminal_test, action

        agent_index_no = depth % state.getNumAgents()
        if agent_index_no == 0:
            return self.max(state, depth, agent_index_no)
        else:
            return self.min(state, depth, agent_index_no)

    # Evaluates the max function of the MINIMAX algorithm.
    def max(self, state, depth, agent_index_no):
        maximum_value = (float("-inf"))
        actions = state.getLegalActions(agent_index_no)
        recommended_action = None

        if len(actions) == 0:
            return self.evaluationFunction(state), None

        for action in actions:
            m_value, m_action = self.recurrence(state.generateSuccessor(agent_index_no, action), depth + 1)
            if m_value > maximum_value:
                maximum_value = m_value
                recommended_action = action

        return maximum_value, recommended_action

    # Evaluates the min function of the MINIMAX algorithm.
    def min(self, state, depth, agent_index_no):
        minimum_value = (float("inf"))
        actions = state.getLegalActions(agent_index_no)
        recommended_action = None

        if len(actions) == 0:
            return self.evaluationFunction(state), None

        for action in actions:
            m_value, m_action = self.recurrence(state.generateSuccessor(agent_index_no, action), depth + 1)
            if m_value < minimum_value:
                minimum_value = m_value
                recommended_action = action

        return minimum_value, recommended_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # Start checking for the optimal move.
        value, action = self.recurrence(gameState, 0, float("-inf"), float("inf"))
        return action

    # Check the terminal state (Is the state a win, lose or max depth i.e the end nodes are reached).
    def is_terminal_state(self, state, depth):
        if state.isWin() or state.isLose() or (depth == self.depth * state.getNumAgents()):
            return self.evaluationFunction(state), None
        else:
            return None, None

    # Using the below method to call the appropriate (max, min) functions with the correct agent index.
    # Pass the alpha beta values used for pruning the sub graphs of the tree.
    def recurrence(self, state, depth, alpha, beta):
        terminal_test, action = self.is_terminal_state(state, depth)
        if terminal_test is not None:
            return terminal_test, action

        agent_index_no = depth % state.getNumAgents()
        if agent_index_no == 0:
            return self.max(state, depth, agent_index_no, alpha, beta)
        else:
            return self.min(state, depth, agent_index_no, alpha, beta)

    # Evaluates the min function of the MINIMAX algorithm.
    def max(self, state, depth, agent_index_no, alpha, beta):
        maximum_value = (float("-inf"))
        actions = state.getLegalActions(agent_index_no)
        recommended_action = None

        if len(actions) == 0:
            return self.evaluationFunction(state), None

        for action in actions:
            m_value, m_action = self.recurrence(state.generateSuccessor(agent_index_no, action), depth + 1, alpha, beta)
            if m_value > maximum_value:
                maximum_value = m_value
                recommended_action = action

            if maximum_value > beta:
                return maximum_value, recommended_action

            alpha = max(alpha, maximum_value)

        return maximum_value, recommended_action

    # Evaluates the min function of the MINIMAX algorithm.
    def min(self, state, depth, agent_index_no, alpha, beta):
        minimum_value = (float("inf"))
        actions = state.getLegalActions(agent_index_no)
        recommended_action = None

        if len(actions) == 0:
            return self.evaluationFunction(state), None

        for action in actions:
            m_value, m_action = self.recurrence(state.generateSuccessor(agent_index_no, action), depth + 1, alpha, beta)
            if m_value < minimum_value:
                minimum_value = m_value
                recommended_action = action

            if minimum_value < alpha:
                return minimum_value, recommended_action

            beta = min(beta, minimum_value)

        return minimum_value, recommended_action

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
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # Start the recurrence procedure, to find the best move.
        value, action = self.recurrence(gameState, 0)
        return action

    # Check the terminal state (Is the state a win, lose or max depth i.e the end nodes are reached).
    def is_terminal_state(self, state, depth):
        if state.isWin() or state.isLose() or (depth == self.depth * state.getNumAgents()):
            return self.evaluationFunction(state), None
        else:
            return None, None

    # Using the below method to call the appropriate (max, min) functions with the correct agent index.
    def recurrence(self, state, depth):
        terminal_test, action = self.is_terminal_state(state, depth)
        if terminal_test is not None:
            return terminal_test, action

        agent_index_no = depth % state.getNumAgents()
        if agent_index_no == 0:
            return self.max(state, depth, agent_index_no)
        elif agent_index_no % 1 == 0:
            return self.expectiminimaxAverage(state, depth, agent_index_no)
        else:
            return self.min(state, depth, agent_index_no)

    # Evaluates the min function of the MINIMAX algorithm.
    def max(self, state, depth, agent_index_no):
        maximum_value = (float("-inf"))
        actions = state.getLegalActions(agent_index_no)
        recommended_action = None

        if len(actions) == 0:
            return self.evaluationFunction(state), None

        for action in actions:
            m_value, m_action = self.recurrence(state.generateSuccessor(agent_index_no, action), depth + 1)
            if m_value > maximum_value:
                maximum_value = m_value
                recommended_action = action

        return maximum_value, recommended_action

    # Evaluates the min function of the MINIMAX algorithm.
    def min(self, state, depth, agent_index_no):
        minimum_value = (float("inf"))
        actions = state.getLegalActions(agent_index_no)
        recommended_action = None

        if len(actions) == 0:
            return self.evaluationFunction(state), None

        for action in actions:
            m_value, m_action = self.recurrence(state.generateSuccessor(agent_index_no, action), depth + 1)
            if m_value < minimum_value:
                minimum_value = m_value
                recommended_action = action

        return minimum_value, recommended_action

    # Evaluates the average function of the EXPECTIMAX algorithm.
    def expectiminimaxAverage(self, state, depth, agent_index_no):
        actions = state.getLegalActions(agent_index_no)

        if len(actions) == 0:
            return self.evaluationFunction(state), None

        total_chance_events = len(actions);
        total_value = 0
        for action in actions:
            m_value, m_action = self.recurrence(state.generateSuccessor(agent_index_no, action), depth + 1)
            total_value += m_value


        return float((total_value))/ total_chance_events, None

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    ################################################################################
    # Pac-man position, the Ghost positions, food positions, power pallet positions,
    # Ghost states for getting the time during which the ghost are scared.
    ################################################################################
    pacman_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()
    food_positions = currentGameState.getFood().asList()
    power_pellets_positions = currentGameState.getCapsules()
    ghost_states = currentGameState.getGhostStates()
    scared_times = [ghostState.scaredTimer for ghostState in ghost_states]

    # The total cost evaluated by the evaluation function.
    total_cost = 0

    # Find the nearest distance to the closest food.
    # Give high priority to eating food after the power pallet.
    nearest_food_distance = float("inf")
    nearest_food_position = None
    for food_position in food_positions:
        # Get all the distances.
        distance = util.manhattanDistance(food_position, pacman_position)

        # Check for the nearest distance.
        if distance < nearest_food_distance:
            nearest_food_distance = distance
            nearest_food_position = food_position

    if len(food_positions) > 0:
        total_cost += (4.0 / mazeDistance(nearest_food_position, pacman_position, currentGameState))

    # If the ghost is near then the value of the evaluation function should be less,
    # else the value of the evaluation function should be more.
    #
    # If the ghost is scared, then direct Pac-man to eat the ghost.
    # Ghost scared time.
    ghost_scared_time = 0
    for scared_time in scared_times:
        ghost_scared_time += scared_time

    # Ghost distance.
    nearest_ghost_distance = float("inf")

    for ghost_position in ghost_positions:
        distance = manhattanDistance(ghost_position, pacman_position)
        if distance < nearest_ghost_distance:
            nearest_ghost_distance = distance

    if nearest_ghost_distance > 0 and ghost_scared_time == 0:
        total_cost += (-2.0/nearest_ghost_distance)

    #
    #  If a power pallet is available then, setting a high negative value to direct Pac-man to eat the power pallet.
    #
    total_cost -= (25 * len(power_pellets_positions))

    return total_cost + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction

############################################################################
# The below code is taken from the Search.py file from the previous project,
# to calculate the exact distance between Pac-man and food pallets.
############################################################################
class PositionSearchProblem:
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # Creating the fringe for holding the the states. The fringe is a FIFO data structure to explore the shallowest node first.
    fringe = util.Queue()
    fringe.push((problem.getStartState(), [], 0))

    visited = []

    while not fringe.isEmpty():
        # Get a new state.
        current_node = fringe.pop()

        # Process the node if not yet visited.
        state, action, cost = current_node
        if state in visited:
            continue

        visited.append(state)

        if problem.isGoalState(state):
            # Return path to Goal.
            return action

        # Put in the FIFO list the successors if not yet visited.
        for child_state, child_action, child_cost in problem.getSuccessors(state):
            if (child_state not in visited):
                fringe.push((child_state, action + [child_action], child_cost))

    return []

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(breadthFirstSearch(prob))
