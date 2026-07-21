import csv
import numpy as np
import matplotlib.pyplot as plt


def compute_mse_simulations(exact_sim: str, approx_sim: str, frames: int, body_count: int):
    mse_list = []
    for i in range(frames):
        sim1 = f"./outputs/{exact_sim}/{exact_sim}_frame_{i}.csv"
        sim2 = f"./outputs/{approx_sim}/{approx_sim}_frame_{i}.csv"

        csvfile = open(sim1, "r", newline="")
        reader = csv.reader(csvfile)
        data1 = np.array(list(reader)[1:body_count+1], dtype=float)
        csvfile.close()

        csvfile = open(sim2, "r", newline="")
        reader = csv.reader(csvfile)
        data2 = np.array(list(reader)[1:body_count+1], dtype=float)
        csvfile.close()

        body_count = len(data2)

        res = np.sum((data1 - data2)**2, axis=1)
        mse = np.sum(res) / body_count
        mse_list.append(mse)
    
    return mse_list


n = 100
m = 10000
mse_list0_05 = compute_mse_simulations("sim5naive", "sim5theta0_05", n, m)
mse_list0_1 = compute_mse_simulations("sim5naive", "sim5theta0_1", n, m)
mse_list0_5 = compute_mse_simulations("sim5naive", "sim5theta0_5", n, m)


plt.plot(range(n), mse_list0_05)
print("1")
plt.plot(range(n), mse_list0_1)
print("2")
plt.plot(range(n), mse_list0_5)
print("3")
plt.grid(True)
plt.legend(["theta=0.05", "theta=0.1", "theta=0.5"])
plt.savefig("./images/mse.pdf")
plt.show()