import copy

from player import Player
import numpy as np


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"
        self.mode = 'rw'

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        # TODO (Additional: Implement roulette wheel here)
        # TODO (Additional: Implement SUS here)

        # TODO (Additional: Learning curve)

        if self.mode == 'rw':
            return self.roulette_wheel(players, num_players)[: num_players]
        elif self.mode == 'sus':
            return self.sus_selector(players, num_players)[: num_players]
        else:
            sorted_players = sorted(players, key=lambda player: player.fitness, reverse=True)
            return sorted_players[: num_players]

    def roulette_wheel(self, players, num_players):
        probas = []
        sum_fitness = 0
        for player in players:
            sum_fitness += player.fitness
        for player in players:
            probas.append(player.fitness / sum_fitness)
        for i in range(1, len(players)):
            probas[i] += probas[i - 1]

        results = []
        randoms = []

        for j in range(num_players):
            random_number = np.random.uniform(0, 1, 1)
            for i, proba in enumerate(probas):
                if random_number <= proba:
                    results.append(self.clone_player(players[i]))
                    break

        return results

    def sus_selector(self, players, num_players):
        probas = []
        sum_fitness = 0
        for player in players:
            sum_fitness += player.fitness
        for player in players:
            probas.append(player.fitness / sum_fitness)
        for i in range(1, len(players)):
            probas[i] += probas[i - 1]

        random_number = np.random.uniform(0, 1 / num_players, 1)
        step = (probas[len(probas) - 1] - random_number) / num_players
        results = []
        # print(f"step {step}")
        # print(f"last : {probas[len(probas) - 1]}")
        # print(f"num_players {num_players}")
        for i in range(num_players):
            now = (i+1) * step
            for i, proba in enumerate(probas):
                if now <= proba:
                    results.append(self.clone_player(players[i]))
                    break
        return results

    def add_noise(self, array, threshold):
        random_number = np.random.uniform(0, 1, 1)
        if random_number < threshold:
            array += np.random.randn(array.shape[0] * array.shape[1]).reshape(array.shape[0], array.shape[1])

    def mutate(self, child):
        # child: an object of class `Player`
        threshold = 0.15

        self.add_noise(child.nn.W_1, threshold)
        self.add_noise(child.nn.W_2, threshold)
        self.add_noise(child.nn.b_1, threshold)
        self.add_noise(child.nn.b_2, threshold)

    def crossover(self, child1, child2, parent1, parent2):
        # print("child1 ", child1.shape)
        row_size, column_size = child1.shape
        section_1, section_2 = int(row_size / 3), int(2 * row_size / 3)

        # parent_chooser = np.random.uniform(0, 1, 1)
        # if parent_chooser > 0.8:
        child1[:section_1, :] = parent1[:section_1, :]
        child1[section_1:section_2, :] = parent2[section_1:section_2, :]
        child1[section_2:, :] = parent1[section_2:, :]

        child2[:section_1, :] = parent2[:section_1, :]
        child2[section_1:section_2, :] = parent1[section_1:section_2, :]
        child2[section_2:, :] = parent2[section_2:, :]
        # else:
        #     child1[:section_1, :] = parent2[:section_1, :]
        #     child1[section_1:section_2, :] = parent1[section_1:section_2, :]
        #     child1[section_2:, :] = parent2[section_2:, :]
        #
        #     child2[:section_1, :] = parent1[:section_1, :]
        #     child2[section_1:section_2, :] = parent2[section_1:section_2, :]
        #     child2[section_2:, :] = parent1[section_2:, :]

        # return child1, child2


    def child_production(self, parent1, parent2):
        child1 = self.clone_player(parent1)
        child2 = self.clone_player(parent2)
        # print("BEFORE", child1.nn.W_1[0])
        # before = child2.nn.b_2.copy()

        self.crossover(child1.nn.W_1, child2.nn.W_1, parent1.nn.W_1, parent2.nn.W_1)
        self.crossover(child1.nn.W_2, child2.nn.W_2, parent1.nn.W_2, parent2.nn.W_2)
        self.crossover(child1.nn.b_1, child2.nn.b_1, parent1.nn.b_1, parent2.nn.b_1)
        self.crossover(child1.nn.b_2, child2.nn.b_2, parent1.nn.b_2, parent2.nn.b_2)
        # print("AFTER", child1.nn.W_1[0])
        # after = child2.nn.b_2.copy()
        # print(after - before)
        self.mutate(child1)
        self.mutate(child2)

        return child1, child2

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:

            return [Player(self.game_mode) for _ in range(num_players)]
        else:

            children = []
            # print(len(prev_players))
            prev_parents = prev_players.copy()
            if self.mode == 'rw':
                prev_parents = self.roulette_wheel(prev_players, len(prev_players))
            elif self.mode == 'sus':
                prev_parents = self.sus_selector(prev_players, len(prev_players))

            # print(len(prev_parents))
            for i in range(0, len(prev_parents), 2):

                child1, child2 = self.child_production(prev_parents[i], prev_parents[i + 1])
                children.append(child1)
                children.append(child2)

            return children

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
