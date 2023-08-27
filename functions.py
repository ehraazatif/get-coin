import neat
from random import randint
from math import sqrt
import pickle
import visualize
import time
import os

class NN:
    def __init__(self, config):
        self.config = config #set config

        #variables that set the size of the grid for the environment
        self.min = 0
        self.max = 9
        return
    
    def gen_inputs(self, player, coin): #returns a tuple for the input of the NN
        return (player[0], player[1], coin[0], coin[1], player[0]-coin[0], player[1]-coin[1])

    def gen_coin_test(self, player=None): #randomly generate coin for testing environment
        coin = [randint(self.min, self.max), randint(self.min, self.max)]
        while coin == player:
            coin = [randint(self.min, self.max), randint(self.min, self.max)]
        return coin
    
    def gen_coin_train(self): #randomly generate coin for training environment (smaller bounds compared to testing environment)
        coin = [randint(-1, 1), randint(-1, 1)]
        while coin == [0, 0]:
            coin = [randint(-1, 1), randint(-1, 1)]
        return coin
    
    def calc_dist(self, a, b): #calculate Euclidean distance between player and coin
        return sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 )
    
    def move_player(self, n, player): #move player based on input
        if n == 0:
            player[0] += 1
        elif n == 1:
            player[0] -= 1
        elif n == 2:
            player[1] += 1
        else:
            player[1] -= 1
        return
    
    def game(self, net, genome): #this is the "game", where the player will try to get to the coin
        player = [0,0] #generate player
        coin = self.gen_coin_train() #generate coin
        moves = 0 #keep track of moves made by player
        run = True

        while run: #main game loop
            init_dist = self.calc_dist(player, coin) #get initial distance between player and coin
            inputs = self.gen_inputs(player, coin) #get inputs
            output = net.activate(inputs) #get output
            neuron = output.index(max(output)) #determine which neuron was activated
            self.move_player(neuron, player) #NN moves the player
            moves += 1 #increment # of moves
            new_dist = self.calc_dist(player, coin) #get ew distance between player and coin

            #increase genome fitness if moved closer to coin; decrease otherwise
            genome.fitness += 1 if new_dist < init_dist else -1

            if moves == 100: #player shouldn't be able to go 100 moves without getting to the coin
                run = False

            if player == coin: #endgame if player gets to coin
                genome.fitness = 5
                run = False
        return
    
    def eval_genomes(self, genomes, config, gen): #fitness function
        for id, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config) #create NN
            self.game(net, genome) #genome plays game
            print('Gen: {}, ID: {}, Fit: {}'.format(gen, id, genome.fitness)) #output genome stats
        return
    
    def train(self): #training function
        p = neat.Population(self.config) #define population
        winner = p.run(self.eval_genomes, 5000) #train for max of 5000 generations
        pickle.dump(winner, open('winner.pickle', 'wb')) #save winning genome
        print(winner)
        visualize.draw_net(self.config, winner) #display NN
        return
    
    def test(self, genome): #testing functions
        player = [randint(self.min, self.max), randint(self.min, self.max)] #init random player
        coin = self.gen_coin_test(player) #init random coin
        moves = 0 #keep track of number of moves
        net = neat.nn.FeedForwardNetwork.create(genome, self.config) #create NN
        run = True

        while run: #main loop
            os.system('cls')
            self.print_game(player, coin) #print the game in console as a grid
            inputs = self.gen_inputs(player, coin) #get inputs
            output = net.activate(inputs) #get outputs
            neuron = output.index(max(output)) #determine which neuron has been activated
            self.move_player(neuron, player) #move player
            moves += 1 #keep track of moves

            if moves == 100: #set limit on number of moves so that we don't start looping endlessly
                print('reached 100 moves')
                run = False

            if player == coin: #display the game one last time, showing that the player has reached the coin
                time.sleep(2)
                os.system('cls')
                self.print_game(player, coin)
                run = False

            time.sleep(2)
        return

    def print_game(self, player, coin): #function to print game environment as a grid
        #init game grid
        game = [
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
        ]

        #set player and coin coordinates in game grid
        game[coin[0]][coin[1]] = 2
        game[player[0]][player[1]] = 1

        #display game; player is represented by @ and coin is represented by O
        print('+'*12)
        for i in range(len(game)):
            print('+', end='')
            for j in range(len(game[i])):
                if game[i][j] == 0:
                    print('-',end='')
                elif game[i][j] == 1:
                    print('@',end='')
                else:
                    print('O',end='')
            print('+')
        print('+'*12)
        return
