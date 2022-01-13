import matplotlib.pyplot as plt


def main():

    plt.figure(figsize=(6, 3), dpi=300)
    grid = plt.GridSpec(2, 4, wspace=0.02, hspace=0.02)
    axm = plt.subplot(grid[0:2, 0:2])
    ax1 = plt.subplot(grid[0, 2:])
    ax2 = plt.subplot(grid[1, 2:])

    print(axm.get_position())
    print(ax1.get_position())
    print(ax2.get_position())


    print()

    plt.figure(figsize=(6, 3), tight_layout=True, dpi=300)
    grid = plt.GridSpec(7, 13, wspace=.02, hspace=0.02)
    axm = plt.subplot(grid[0:5, 0:5])
    ax1 = plt.subplot(grid[0:2, 6:11])
    ax2 = plt.subplot(grid[3:5, 6:11])
    axc = plt.subplot(grid[6:7, 0:])
    axg = plt.subplot(grid[0:5, 12:13])
    plt.suptitle('Figure 2')

    print(axm.get_position())
    print(ax1.get_position())
    print(ax2.get_position())
    print(axc.get_position())
    print(axg.get_position())

    plt.show()


if __name__ == '__main__':
    main()