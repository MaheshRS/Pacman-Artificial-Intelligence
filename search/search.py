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
    #util.raiseNotDefined()
    # Creating the fringe for holding the the states. The fringe is a LIFO data structure to explore the deepest node first.
    fringe = util.Stack()
    explored = set()
    startState = problem.getStartState()

    # If the start state is a goal state then do not return any actions.
    # Else push the start state into the fringe and initialize the action list to empty, since the action will be added once the nodes are expanded.
    if problem.isGoalState(startState):
        return []
    else:
        fringe.push((startState,[]))

    while not fringe.isEmpty():
        
        # Check if the current state is a goal state. If Yes return the actions.
        state, action_trail = fringe.pop()
        if(problem.isGoalState(state)):
            return action_trail

        # Add the state that is being expanded into explored.
        explored.add(state)

        # Adding the successors into the fringe.
        successors = problem.getSuccessors(state);
        for node in successors:
            child_state, action, cost = node

            if (child_state not in explored):
                fringe.push((child_state, action_trail + [action]))

    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
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

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    # Creating the fringe for holding the the states. The fringe is a Priority Queue data structure to explore the least cost node first.
    fringe = util.PriorityQueue()
    visited = []

    # If the start state is a goal state then do not return any actions.
    if problem.isGoalState(problem.getStartState()):
        return []
    else:
        fringe.push((problem.getStartState(), [], 0), 0)

    while not fringe.isEmpty():
        # Get a new state.
        state, action, cost = fringe.pop()

        # Process the node if not yet visited.
        if state in visited:
            continue

        visited.append(state)

        if problem.isGoalState(state):
            # Return path to the goal.
            return action
        else:
            # Put in the queue the successors if not yet visited.
            for child_state, child_action, child_cost in problem.getSuccessors(state):
                if (child_state not in visited):
                    total_cost = cost + child_cost;
                    fringe.push((child_state, action + [child_action], total_cost), total_cost)
                    
    return None

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    # Creating the fringe for holding the the states. The fringe is a Priority Queue data structure to get the least cost path.
    fringe = util.PriorityQueue()
    rootEstimatedCost = heuristic(problem.getStartState(), problem)
    visited = []

    # If the start state is a goal state then do not return any actions.
    if problem.isGoalState(problem.getStartState()):
        return []
    else:
        fringe.push((problem.getStartState(), [], 0), rootEstimatedCost)

    while not fringe.isEmpty():
        # Get a new state.
        state, action, cost = fringe.pop()

        # Process the node if not yet visited.
        if state in visited:
            continue

        visited.append(state)

        if problem.isGoalState(state):
            # Return path to here.
            return action
        else:
            # Put in the queue the successors if not yet visited.
            for child_state, child_action, child_cost in problem.getSuccessors(state):
                if (child_state not in visited):
                    estimatedCost = heuristic(child_state, problem)
                    fringe.push((child_state, action + [child_action], cost + child_cost), (cost + child_cost + estimatedCost))
                    
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
