import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
import pandas as pd

with open("return - Copy (5).pckl", "rb") as f:
    returns = pickle.load(f)

with open("return - Copy (6).pckl", "rb") as f:
    returns1 = pickle.load(f)

minlen = min(len(returns), len(returns1))
full_ret = np.stack((returns[:minlen],returns[:minlen], returns1[:minlen]))

df = pd.DataFrame(full_ret).melt()
ax = sns.lineplot(x="variable", y="value", data=df, err_style='bars')

# timesteps = np.arange(0, len(returns)*20, 20)
# # Plot the responses for different events and regions
# ax = sns.lineplot(x=timesteps, y=returns)
ax.set(xlabel='Iterations', ylabel='Loss')
plt.show()
