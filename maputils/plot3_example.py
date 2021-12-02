#!/usr/bin/env python3

import matplotlib.pyplot as plt

def main(azim=None, dist=None, elev=None, projection='3d'):
    # azim, dist, elev

    import sys

    # import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

    fig = plt.figure()
    ax = fig.gca(projection=projection)

    # if len(sys.argv) > 1:
    #     azim = int(sys.argv[1])
    # else:
    #     azim = None
    # if len(sys.argv) > 2:
    #     dist = int(sys.argv[2])
    # else:
    #     dist = None
    # if len(sys.argv) > 3:
    #     elev = int(sys.argv[3])
    # else:
    #     elev = None

    # Make data.
    X = np.arange(-5, 6, 1)
    Y = np.arange(-5, 6, 1)
    X, Y = np.meshgrid(X, Y)
    Z = X**2

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)

    # Labels.
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if azim is not None:
        ax.azim = azim
    if dist is not None:
        ax.dist = dist
    if elev is not None:
        ax.elev = elev

    print('ax.azim = {}'.format(ax.azim))
    print('ax.dist = {}'.format(ax.dist))
    print('ax.elev = {}'.format(ax.elev))
    ax.set_title('azim: {} | dist: {} | elev: {}'.format(ax.azim, ax.dist, ax.elev))

    plt.savefig(
        'main_{}_{}_{}.png'.format(ax.azim, ax.dist, ax.elev),
        format='png',
        bbox_inches='tight'
    )

    plt.draw()

if __name__ == '__main__':
    main()
    main(azim=-90, dist=10, elev=0) # looking at x,z axis (x labels on bottom, z labels on right)
    plt.show()