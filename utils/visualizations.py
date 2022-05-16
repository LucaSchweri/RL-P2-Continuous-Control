import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(scores, names, ma=100, show=False):
    """Plots the score over the episodes
    
    Params
    ======
        scores (list): list of scores
        names (list): names of the methods used
        ma (int): moving average window
        show (bool): whether the figure should be displayed or saved as image
    """
    
    fig = plt.figure(figsize=(14, 10))
    for score, name in zip(scores, names):
        moving_average = np.convolve(score, np.ones(ma), 'valid') / ma
        if len(names) == 1:
            plt.plot(score, label=name)
            plt.plot([i for i in range(ma-1, len(score))], moving_average, label=f"mean({name})")
        else:
            plt.plot([i for i in range(ma-1, len(score))], moving_average, label=name)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    if len(names) == 1 and not show:
        plt.savefig(f"./data/{names[0]}/learning_curve.png")
    else:
        plt.show()
    plt.close(fig)


def plot_learning_curve2(scores, names, ma=100, show=False):
    """Plots the loss over the episodes

    Params
    ======
        scores (list): list of losses
        names (list): names of the methods used
        ma (int): moving average window
        show (bool): whether the figure should be displayed or saved as image
    """

    fig = plt.figure(figsize=(14, 10))
    for score, name in zip(scores, names):
        moving_average = np.convolve(score, np.ones(ma), 'valid') / ma
        if len(names) == 1:
            plt.plot(score, label=name)
            plt.plot([i for i in range(ma - 1, len(score))], moving_average, label=f"mean({name})")
        else:
            plt.plot([i for i in range(ma - 1, len(score))], moving_average, label=name)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    if len(names) == 1 and not show:
        plt.savefig(f"./data/{names[0]}/learning_curve2.png")
    else:
        plt.show()
    plt.close(fig)