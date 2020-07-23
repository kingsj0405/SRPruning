def visualize_both(name, norm):
    fig, ax = plt.subplots(1, 2)
    ax[0].set_xscale('linear')
    visualize_norm_stat(f"{name}_lin", norm, ax[0])
    ax[1].set_xscale('log')
    visualize_norm_stat(f"{name}_log", norm.abs(), ax[1])
    plt.show()
    #plt.savefig(f"images/{name}.png", dpi=300)

def visualize_norm_stat(name, norm, ax):
    # Draw each point
    y = np.zeros(np.shape(norm))
    ax.plot(norm, y, '|')
    estimator = stats.gaussian_kde(norm, bw_method='silverman')
    # Draw kernel density estimate
    X = np.arange(norm.min() * 1.1, norm.max() * 1.1, 0.1)
    K = estimator(X)
    ax.plot(X, K, label=f'{name}')
    # Set other things
    ax.legend(loc='best')