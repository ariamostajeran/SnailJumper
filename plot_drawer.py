import matplotlib.pyplot as plt
f = open("data.txt", "r")
bests = []
means = []
worsts = []
for line in f:
    best, worst, mean = line.split()
    bests.append(float(best))
    worsts.append(float(worst))
    means.append(float(mean))
# y = range(len(bests))
plt.plot(bests, label="bests")
plt.plot(means, label="means")
plt.plot(worsts, label="worsts")
plt.xlabel("Generations")
plt.ylabel("Score")
plt.legend()
plt.savefig("25gen_35seed.png")
