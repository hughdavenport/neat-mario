from __future__ import print_function
import neat
import pickle       # pip install cloudpickle
import os
import sys

from SuperMarioBrosNes.game import SuperMarioBros

import visualize

from utilities import saveNet, isFocussed

from trackers import StagnationTracker

training = True
rounding = True

debug = False

def simulateGame(key, net):
    game = SuperMarioBros()

    last_change = 0
    fitness = 0
    while not game.isFinished():
        if debug:
            game.render()
            if not isFocussed():
                continue
        # Round val, so don't need to get 1. and 0. exact
        output = net.activate(game.state())
        if rounding:
            output = [round(val) for val in output]
        game.step(output)

        if game.fitness() > fitness:
            fitness = game.fitness()
            last_change = 0
        else:
            last_change += 1
            if last_change > 250: # Around 10seconds in game
                break
            if last_change > 125: # I guess around 5 seconds in game?
                break

    if training:
        # At this point, if fitness is *stuck*, then train off some data instead
        #fitness += old_training_error(net, game)
        fitness += training_error(net, game)

    game.close()
    return fitness

def training_error(net, game):
    fitness = game.fitness()
    training_filename = "training-{}.csv".format(fitness)
    if not os.path.isfile(training_filename):
        return 0.

    with open(training_filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            arr = list(map(float, line.rstrip()[1:-1].split('","')))
            state, expected = arr[:-5], arr[-5:]
            output = net.activate(state)
            error += sum([(expected[i] - output[i])**2 for i in range(0, len(output))]) / len(output)
        error /= len(lines)
    return 1. - error

def old_training_error(net, game):
    fitness = game.fitness()
    if fitness == 40:
        # Haven't moved, train a bit more on moving right and not left
        output = net.activate(game.state())
        return 1. - ((1. - output[SuperMarioBros.KEYS["RIGHT"]])**2 + (0. - output[SuperMarioBros.KEYS["LEFT"]])**2) / 2.

    if fitness == 297:
        # Run into first enemy, try jumping
        error = 0.
        with open("training-297", "r") as f:
            lines = f.readlines()
            for line in lines:
                arr = list(map(float, line.rstrip()[1:-1].split('","')))
                state, expected = arr[:-5], arr[-5:]
                output = net.activate(state)
                error += sum([(expected[i] - output[i])**2 for i in range(0, len(output))]) / len(output)
            error /= len(lines)
        return 1. - error

    # FIXME: do this as a tracker and save every X gens of no improvement
    if fitness == 434:
        # First pipe, need to jump over it
        saveNet(net, "stuck-434.net")
        pass

    if fitness == 723:
        # A few pipes later maybe?
        saveNet(net, "stuck-723.net")
        pass

    return 0.

def eval_genome(genome, config):
    genome.fitness = 0.0
    if config.genome_config.feed_forward:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        net = neat.nn.RecurrentNetwork.create(genome, config)

    genome.fitness = simulateGame(genome.key, net)

    return genome.fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run(config_file, restore_file=None):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = None
    if restore_file is None:
        p = neat.Population(config)
    else:
        p = neat.Checkpointer.restore_checkpoint(restore_file)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=10, time_interval_seconds=None))

    p.add_reporter(StagnationTracker())

    # Run for up to 300 generations.
    winner = None
    if debug:
        winner = p.run(eval_genomes, 300)
    else:
        pe = neat.ParallelEvaluator(4, eval_genome)
        winner = p.run(pe.evaluate)

    print('\nBest genome:\n{!s}'.format(winner))

    print("Saving as winner-genome-{}.pkl".format(winner.fitness))
    with open('winner-genome-{}.pkl'.format(winner.fitness), 'wb') as output:
        pickle.dump(winner, output, 1)

    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.

    # TODO have human/replay mode

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    # TODO parse sysargs properly
    if len(sys.argv) >= 2:
        run(config_path, sys.argv[1])
    else:
        run(config_path)
