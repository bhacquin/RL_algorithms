from __future__ import division

import collections
import gym
import numpy as np
from copy import deepcopy, copy
import time
import math
import random




class treeNode():                          #CLASSE NOEUD, ASSOCIE UN ETAT DE L'ENVIRONNEMENT A CHAQUE NOEUD DE L'ARBRE
    def __init__(self, env, reward, terminal, parent):
        self.state = env
        self.obs = env.state
        self.isTerminal = terminal
        self.isFullyExpanded = terminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}
        self.reward = reward



class MCTS():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2), use_mean=True):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.use_mean = use_mean

    def search(self, initialState):                  #LA FONCTION QUI VA RENDRE L'ACTION A CHOISIR
        self.root = treeNode(initialState, 0, False, None)            #ON CREE LA RACINE DE L'ARBRE AVEC L'ENVIRONNEMENT DONNE

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)  #ON CHOISIT LA MEILLEURE ACTION (SANS CONSTANTE D'EXPLORATION CAR C'EST CELLE QUE L'AGENT VA EFFECTIVEMENT CHOISIR)
        return self.getAction(self.root, bestChild)

    def executeRound(self):                          #C'EST CE QUI SE PASSE A CHAQUE ITERATION : SELECTION - EXPANSION - SIMULATION - BACKPROPAGATION
        node = self.selectNode(self.root)
        reward = self.randomPolicy(node)
        self.backpropogate(node, reward)

    def selectNode(self, node):                      #SELECTION : ON TROUVE UN NOEUD QUI N'A PAS ETE ENTIEREMENT ETENDU ET ON L'ETEND
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):                         #EXPANSION : ON EXPAND LE NOEUD CHOISIT
        actions = self.getPossibleActions(node.state)
        for action in actions:
            if action not in node.children:
                env, reward, terminal = self.takeAction(node.state, action)
                newNode = treeNode(env, reward, terminal, node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):          #ON FAIT REMONTER LA REWARD A TOUS LES PARENTS ET ON LEUR AJOUTE UNE VISITE JUSQU'A ATTEINDRE LA RACINE
        while node is not None:
            node.numVisits += 1
            if self.use_mean:
                node.totalReward += reward
            else:
                node.totalReward = max(node.totalReward, reward)
            node = node.parent

    def getBestChild(self, node, explorationValue): #LORS DE LA SELECTION : QUEL NOEUD FILLE ON CHOISIT
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            if self.use_mean:
                nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
                    2 * math.log(node.numVisits) / child.numVisits)          #DILEMME EXPLORATION/EXPLOITATION (VOIR "UPPER CONFIDENCE BOUND FOR TREES")
            else:
                nodeValue = child.totalReward + explorationValue * math.sqrt(
                    2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def getAction(self, root, bestChild):               #DONNE L'ACTION DE LA MEILLEURE FILLE
        for action, node in root.children.items():
            if node is bestChild:
                return action

    def randomPolicy(self, node):                       #SIMULATION : LE NOEUD SELECTIONNE EST SIMULE ET ON CALCULE LA REWARD ASSOCIEE EN FAISANT DES CHOIX ALEATOIRES
        tot_r = 0
        terminal = node.isTerminal
        env = node.state
        env_copy = deepcopy(env)
        while not terminal:
            try:
                action = random.choice(self.getPossibleActions(env))
            except IndexError:
                raise Exception("Non-terminal state has no possible actions")
            _, reward, terminal, _ = env_copy.step(action)
            tot_r += reward
        return tot_r

    def getPossibleActions(self, env):                  #TOUTES LES ACTIONS POSSIBLES POUR UN CERTAIN ETAT DE L'ENVIRONNEMENT
        list_actions = []
        for i in range(env.action_space.n):
            list_actions += [i]
        return list_actions

    def takeAction(self, env, action):                  #ON AGIT SUR L'ENVIRONNEMENT EN S'ASSURANT QU'ON LAISSE DANS CHAQUE NOEUD L'ENVIRONNEMENT DANS L'ETAT DU NOEUD
        new_env = deepcopy(env)
        step = new_env.step(action)
        obs = step[0]
        reward = step[1]
        terminal = step[2]

        return new_env, reward, terminal














