import matplotlib.pyplot as plt
import time
from subprocess import Popen, PIPE
def run(exec, entrada):
    start = time.perf_counter()
    command = "./" + exec + " < " + entrada
    process = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = process.communicate()
    end = time.perf_counter()
    return end - start

def run_all(exec, entradas):
    times = []
    for entrada in entradas:
        times.append(run(exec, entrada))
    return times

entradas = ["entradas/in-{}.txt".format(i) for i in range(150)]

# busca local sequencial
times_busca_local_sequencial = run_all("busca-local-sequencial", entradas)

# plot sequencial
# def plot(times, label):
#     plt.plot(times, label=label)
#     plt.xlabel("Entrada")
#     plt.ylabel("Tempo (s)")
#     plt.legend()
#     plt.show()

#plot(times_busca_local_sequencial, "Busca Local Sequencial")

print("x =", times_busca_local_sequencial)
