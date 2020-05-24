from __future__ import print_function
import neat
import pickle       # pip install cloudpickle
import os
import sys

from SuperMarioBrosNes.game import SuperMarioBros

from pynput import keyboard
import time

from utilities import isFocussed

# FIXME this shouldn't be using neat, was just experiment to do live play
debug=True

        #   b  down  left  right  a
pressed = [0.,  0.,   0.,   0.,   0.]


def on_press(key):
    global pressed

    if not isFocussed():
        return

    if key == keyboard.Key.ctrl:
        pressed[SuperMarioBros.KEYS["A"]] = 1.
    if key == keyboard.Key.alt:
        pressed[SuperMarioBros.KEYS["B"]] = 1.
    if key == keyboard.Key.down:
        pressed[SuperMarioBros.KEYS["DOWN"]] = 1.
    if key == keyboard.Key.left:
        pressed[SuperMarioBros.KEYS["LEFT"]] = 1.
    if key == keyboard.Key.right:
        pressed[SuperMarioBros.KEYS["RIGHT"]] = 1.

def on_release(key):
    global pressed

    if not isFocussed():
        return

    if key == keyboard.Key.ctrl:
        pressed[SuperMarioBros.KEYS["A"]] = 0.
    if key == keyboard.Key.alt:
        pressed[SuperMarioBros.KEYS["B"]] = 0.
    if key == keyboard.Key.down:
        pressed[SuperMarioBros.KEYS["DOWN"]] = 0.
    if key == keyboard.Key.left:
        pressed[SuperMarioBros.KEYS["LEFT"]] = 0.
    if key == keyboard.Key.right:
        pressed[SuperMarioBros.KEYS["RIGHT"]] = 0.

listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
listener.start()

def simulateGame(net):
    game = SuperMarioBros()

    with open("training", "w") as f:
        while not game.isFinished():
            game.render()
            if isFocussed():
                if 0.75 in game.state() and game.fitness() <= 297:
                    f.write('"' + '","'.join(map(str, list(game.state()) + pressed)) + '"\n')
                game.step(pressed)
                time.sleep(0.0175)

    game.close()

    return game.fitness()

def eval_genome(genome, config):
    genome.fitness = 0.0
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    genome.fitness = simulateGame(net)

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
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=10, time_interval_seconds=None))

    # Run for up to 300 generations.
    winner = None
    if debug:
        winner = p.run(eval_genomes, 300)
    else:
        pe = neat.ParallelEvaluator(4, eval_genome)
        winner = p.run(pe.evaluate)

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
