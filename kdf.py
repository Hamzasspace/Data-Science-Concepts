import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# dataset
x = [1.33, 0.3, 0.97, 1.1, 0.1, 1.4, 0.4]

# dummy dataset
dataset = np.array([1.33, 0.3, 0.97, 1.1, 0.1, 1.4, 0.4])

# Seaborn KDE PLot
sns.kdeplot(data=dataset, 
            bw_adjust=0.3,
            linewidth=2.5, fill=True)
plt.title("Seaborn KDE PLot", fontsize=16)
plt.xlabel("Values", fontsize=12)
plt.ylabel("Density", fontsize=12)


plt.show()