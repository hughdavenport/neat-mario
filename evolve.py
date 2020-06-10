from __future__ import print_function
import neat
import pickle       # pip install cloudpickle
import os
import sys

from SuperMarioBrosNes.game import SuperMarioBros

import visualize

from utilities import isFocussed

import trackers

import random

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

        if not game.isControllable():
            game.step()
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
        fitness += training_error(net, fitness)

    game.close()
    return fitness

training_used = []
def training_error(net, fitness):
    global training_used

    generic_training_filename = "training.csv"

    if os.path.isfile(generic_training_filename):
        if generic_training_filename not in training_used:
            print("Using global training file {}".format(generic_training_filename))
            training_used.append(generic_training_filename)
        return _training_error(net, generic_training_filename)

    # Try pixels on the two tiles either side
    for fitness in range(fitness - 16, fitness + 16 + 1):
        training_filename = "training-{}.csv".format(fitness)
        if not os.path.isfile(training_filename):
            if training_filename in training_used:
                print("Removing deleted traing file: ", training_filename)
                training_used.remove(training_filename)
            continue

        if training_filename not in training_used:
            print("Found new training file: ", training_filename)
            training_used.append(training_filename)

        return _training_error(net, training_filename)

    return 0.

def _training_error(net, training_filename):
    best = 1.
    error = 0.
    with open(training_filename, "r") as f:
        lines = f.readlines()
        if lines:
            count = 0
            failures = 0
            for line in lines:
                if line.rstrip() == "===":
                    if failures != count:
                        error /= (count - failures)
                        best = min(best, error)
                    error = 0.
                    count = 0
                    failures = 0
                    continue

                arr = None
                try:
                    arr = list(map(float, line.rstrip()[1:-1].split('","')))
                except ValueError as e:
                    print("Failed parsing line")
                    print(lines.index(line) + 1, line)
                    failures += 1
                    continue
                state, expected = arr[:-5], arr[-5:]
                state = [random.random() if val == 999. else val for val in state]
                output = None
                try:
                    output = net.activate(state)
                except RuntimeError as e:
                    print("Not enough inputs in line")
                    print(lines.index(line) + 1, line)
                    continue
                error += sum([0. if expected[i] == 999. else (expected[i] - output[i])**2 for i in range(0, len(output))]) / (len(output) - expected.count(999.))
                count += 1

            if failures == count:
                return 1. - best

            error /= (count - failures)
            best = min(best, error)

    return 1. - best


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

    p.add_reporter(neat.Checkpointer(generation_interval=10, time_interval_seconds=None))

    p.add_reporter(trackers.AssistanceRequestTracker(p.generation))
    p.add_reporter(trackers.ReportBestTracker(p.generation))

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
