import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

win_list = np.genfromtxt('win_list.csv', delimiter=',')
num_eps = np.genfromtxt('valid_rate.csv', delimiter=',').shape[0]

win_loss_list = np.zeros([num_eps])

for x in win_list:
	win_loss_list[int(x) + 1] = 1

moving_average = 50

plt.title("Win Rate: {}/{}".format(int(sum(win_loss_list)), num_eps - 1))
plot_1 = sns.lineplot(data = np.convolve(win_loss_list, np.ones(moving_average)/moving_average, mode='valid'))
plt.ylabel("Win Rate")
plt.xlabel("Number of episodes")
plt.savefig('win_rate.pdf', 
    bbox_inches = 'tight')
plt.close()
plt.show()  