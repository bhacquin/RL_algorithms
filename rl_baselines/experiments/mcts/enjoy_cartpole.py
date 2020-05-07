import gym

from rl_baselines.baselines import MCTS




env0 = gym.make("CartPole-v0")
env0.seed(1)
env0.reset()

envrend = gym.make("CartPole-v0")
envrend.seed(1)
envrend.reset()

#IL FAUT CREER DEUX ENVIRONNEMENTS EN PARALLELE, UN POUR LE RENDER ET UN POUR LE MCTS. SINON LES DEEPCOPY FONT BUGGER LE env.render()


MCTS = MCTS(iterationLimit = 10, use_mean = True)
 
for i in range(200):
    bestAction = MCTS.search(initialState = env0)
    _, _, done, _ = env0.step(bestAction)
    envrend.step(bestAction)
    envrend.render()
    if i == 199:
        print("GAGNE")
    if done and i != 199:
        print("PERDU")
        print("SCORE : {}/200".format(i))
        break


envrend.close()
