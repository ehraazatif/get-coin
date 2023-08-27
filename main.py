from functions import *

file = 'config.txt' #get config
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, file)
n = NN(config)
n.train() #train
genome = pickle.load(open('winner.pickle', 'rb')) #get winner
n.test(genome) #test