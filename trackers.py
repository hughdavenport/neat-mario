from neat.reporting import BaseReporter
from neat.nn import FeedForwardNetwork, RecurrentNetwork

from utilities import saveNet

class StagnationTracker(BaseReporter):

    def __init__(self):
        self._generation_count = 0
        self._best_net = None
        self._best_fitness = None
        self._best_generation = 0
        self._best_id = None

    def post_evaluate(self, config, population, species, best_genome):
        self._generation_count += 1
        if self._best_net is None or best_genome.fitness > self._best_fitness:
            if config.genome_config.feed_forward:
                self._best_net = FeedForwardNetwork.create(best_genome, config)
            else:
                self._best_net = RecurrentNetwork.create(best_genome, config)
            self._best_fitness = best_genome.fitness
            self._best_generation = self._generation_count
            self._best_id = best_genome.key

        # FIXME: hard coded value
        if self._generation_count - self._best_generation > 50:
            print("Need training assistance, id is", self._best_id, "fitness is", self._best_fitness, "last improvment in generation", self._best_generation)
            saveNet(self._best_net, "need-training-{}.net".format(self._best_fitness))
