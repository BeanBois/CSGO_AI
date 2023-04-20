import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Generate some data
data = np.load('rewards.npy')
match_history = [1 if  x >= 0.9 else 0 for x in data]
color = [1 if x == 1 else 0 for x in match_history]
y = np.ones(len(data))
x = np.arange(len(data))
# Define the colormap and boundaries
cmap = ListedColormap(['red', 'blue'])
norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

# Create the scatter plot
plt.scatter(x, y, c=color, cmap=cmap, norm=norm, s=10)

plt.xlim(min(x)-2, max(x)+2)
plt.ylim(min(y)-2, max(y)+2)

# Create the color bar
cbar = plt.colorbar()
cbar.set_ticklabels(['Enemy Won', '', 'Agent Won'])
# Show the plot
plt.show()
# import matplotlib.pyplot as plt
# import numpy as np
# # Sample policy loss data
# policy_loss = np.zeros(30)

# # Define the x-axis as the number of epochs
# epochs = range(1, len(policy_loss)+1)

# # Create the plot
# plt.plot(epochs, policy_loss, 'bo-', label='Policy Loss')

# # Add labels and title
# plt.xlabel('Epoch')
# plt.ylabel('Policy Loss')
# plt.title('Policy Loss over Epochs')

# # Add legend
# plt.legend(loc='best')

# # Show the plot
# plt.show()