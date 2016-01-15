# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    self.enemyfood = self.getFood(gameState)
    self.allyfood = self.getFoodYouAreDefending(gameState)
    self.myTeam = self.getTeam(gameState)
    self.enemyTeam = self.getOpponents(gameState)

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

  def chooseAction(self, gameState):
    """
  Picks among actions randomly.
  """
    actions = gameState.getLegalActions(self.index)

    '''
  You should change this in your own agent.
  '''
    optimalAction = "Stop"
    danger = False
    threshold = 8
    for enemy in self.enemyTeam:
      if CaptureAgent.getMazeDistance(self, getAgentDistances(enemy), gameState.getAgentPosition(self.index)) < threshold:
        danger = True

    if danger == True:
      optimalAction = random.choice(actions)
    else:
      optimal = 999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        minDistance = min([(self.getMazeDistance(myPos, food), food) for food in foodList])
        food = minDistance[1]

        distance = CaptureAgent.getMazeDistance(self, successor.getAgentPosition(self.index), food)
        if distance < optimal:
          optimal = distance
          optimalAction = action

    return optimalAction
