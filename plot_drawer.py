sorted_players = sorted(players, key=lambda player: player.fitness, reverse=True)
best_fitness = sorted_players[0].fitness
worst_fitness = sorted_players[len(sorted_players) - 1].fitness
fitnesses = [player.fitness for player in players]
mean_fitness = sum(fitnesses) / len(fitnesses)
f = open("data.txt", 'a')
f.write(f"{best_fitness} {worst_fitness} {mean_fitness} \n")
