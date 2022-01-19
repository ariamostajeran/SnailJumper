import copy

from player import Player
import numpy as np


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

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
        player_fitness_dict = {}
        for player in players:
            player_fitness_dict[player] = player.fitness

        sorted_dict = {k: v for k, v in sorted(player_fitness_dict.items(), key=lambda item: item[1])}
        players = list(sorted_dict.keys())
        return players[: num_players]

    def add_noise(self, array, threshold):
        random_number = np.random.uniform(0, 1, 1)
        if random_number < threshold:
            array += np.random.randn(array.shape[0] * array.shape[1]).reshape(array.shape[0], array.shape[1])

    def mutate(self, child):
        # child: an object of class `Player`
        threshold = 0.2

        self.add_noise(child.nn.W_1, threshold)
        self.add_noise(child.nn.W_2, threshold)
        self.add_noise(child.nn.b_1, threshold)
        self.add_noise(child.nn.b_2, threshold)

    def crossover(self, child1, child2, parent1, parent2):
        row_size, column_size = child1.shape
        section_1, section_2 = int(row_size / 4), int(3 * row_size / 4)

        random_number = np.random.uniform(0, 1, 1)

        if random_number > 0.5:
            child1[:section_1, :] = parent1[:section_1:, :]
            child1[section_1:section_2, :] = parent2[section_1:section_2, :]
            child1[section_2:, :] = parent1[section_2:, :]

            child2[:section_1, :] = parent2[:section_1:, :]
            child2[section_1:section_2, :] = parent1[section_1:section_2, :]
            child2[section_2:, :] = parent2[section_2:, :]
        else:
            child1[:section_1, :] = parent2[:section_1:, :]
            child1[section_1:section_2, :] = parent1[section_1:section_2, :]
            child1[section_2:, :] = parent2[section_2:, :]

            child2[:section_1, :] = parent1[:section_1:, :]
            child2[section_1:section_2, :] = parent2[section_1:section_2, :]
            child2[section_2:, :] = parent1[section_2:, :]

    def child_production(self, parent1, parent2):
        child1 = Player(self.game_mode)
        child2 = Player(self.game_mode)

        self.crossover(child1.nn.W_1, child2.nn.W_1, parent1.nn.W_1, parent2.nn.W_1)
        self.crossover(child1.nn.W_2, child2.nn.W_2, parent1.nn.W_2, parent2.nn.W_2)
        self.crossover(child1.nn.b_1, child2.nn.b_1, parent1.nn.b_1, parent2.nn.b_1)
        self.crossover(child1.nn.b_2, child2.nn.b_2, parent1.nn.b_2, parent2.nn.b_2)

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
            # print(f"num_players {num_players}")
            # print(f"prev_players len {len(prev_players)}")

            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # print(f"num_players {num_players}")
            # print(f"prev_players len {len(prev_players)}")
            new_players = prev_players
            children = []

            for i in range(0, len(prev_players), 2):
                child1, child2 = self.child_production(prev_players[i], prev_players[i + 1])
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
