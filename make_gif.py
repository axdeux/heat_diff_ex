import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def make_gif(matrices, mask, max_temp=1, path="heat/", save_name="temp"):
    # print("----------Making gif----------")
    # Open image and read from numpy array
    time_steps = matrices.shape[0]
    
    fig, ax = plt.subplots()
    im = ax.imshow(matrices[0, :, :], vmin=0, vmax=max_temp, animated=True)
    ax.imshow(mask, cmap="gray", alpha=1)
    fig.colorbar(im, ax=ax, label="Temperature (K)")

    # title
    ax.set_title("Heat diffusion | t = 0")

    def update(i):
        im.set_data(matrices[i, :, :])
        ax.set_title("Heat diffusion | t = " + str(i))
        return im
    
    ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=100, repeat=False)

    if not os.path.exists(path):
        os.makedirs(path)

    ani.save(path + save_name + ".gif")