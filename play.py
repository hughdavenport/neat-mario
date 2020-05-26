from __future__ import print_function
from neat import Checkpointer
from neat.nn import FeedForwardNetwork, RecurrentNetwork
import sys
import os

import gzip
try:
    import cPickle as pickle
except ImportError:
    import pickle

from SuperMarioBrosNes.game import SuperMarioBros

from pynput import keyboard
import time

from utilities import isFocussed

debug=True

rounding = True

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

def simulateGame(net=None):
    game = SuperMarioBros()

    answer = input("Create training file? (y/N)")
    if answer.upper() == "Y":
        ai_fitness = 0
        training_filename = "training.csv"
        if net is not None:
            last_change = 0
            while not game.isFinished():
                output = net.activate(game.state())
                if rounding:
                    output = [round(val) for val in output]
                game.step(output)
                if game.fitness() > ai_fitness:
                    ai_fitness = game.fitness()
                    last_change = 0
                else:
                    last_change += 1
                    if last_change > 250: # Around 10seconds in game
                        break
                    if last_change > 125: # I guess around 5 seconds in game?
                        break
            game.close()
            game = SuperMarioBros()

            training_filename = "training-{}.csv".format(ai_fitness)
            print("AI Fitness: ", ai_fitness)

        if os.path.isfile(training_filename):
            answer = input("{} already exists, are you sure? (y/N)".format(training_filename))

        with open(training_filename, "w") as f:
            while f.tell() == 0: # Haven't written anything to train
                run(game, net, f)

                if f.tell() == 0:
                    print("Haven't got anything to train against yet")
                    game.reset()
                elif game.fitness() <= ai_fitness:
                    print("Haven't actually improved yet...")
                    f.seek(0, 0)
                    game.reset()
    else:
        run(game, net)

    game.close()

    return game.fitness()

def run(game, net, training_file=None):
    pause_time = 0.0175

    while not game.isFinished():
        game.render()
        if pressed.count(0.) == len(pressed) and net is not None:
            output = net.activate(game.state())
            if rounding:
                output = [round(val) for val in output]
            game.step(output)
            time.sleep(pause_time / 4)
        else:
            if isFocussed():
                if training_file is not None:
                    training_file.write('"' + '","'.join(map(str, list(game.state()) + pressed)) + '"\n')
                game.step(pressed)
                time.sleep(pause_time)

if __name__ == '__main__':
    # TODO parse sysargs properly
    net = None
    if len(sys.argv) >= 3:
        p = Checkpointer.restore_checkpoint(sys.argv[1])
        genome = p.population[int(sys.argv[2])]
        if p.config.genome_config.feed_forward:
            net = FeedForwardNetwork.create(genome, p.config)
        else:
            net = RecurrentNetwork.create(genome, p.config)
    elif len(sys.argv) >= 2:
        with gzip.open(sys.argv[1]) as f:
            net = pickle.load(f)
        if type(net) == tuple:
            p = Checkpointer.restore_checkpoint(sys.argv[1])
            for key in p.population:
                print(key)
            print("No id selected")
            sys.exit(0)
    print("Fitness:", simulateGame(net))
