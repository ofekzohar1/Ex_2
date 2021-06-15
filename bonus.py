import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans


MAX_K = 10
FIG_FILE_PIC = "elbow.png"  # File saved name


# Using KMeans++ algorithm with random state 0 is needed on the data from iris
# Inspired by predictivehacks.com
# Using save_plot function
def main():
    ds_iris = datasets.load_iris()
    df_iris = ds_iris.data
    inertia_list = []
    for k in range(1, MAX_K + 1):
        model = KMeans(n_clusters=k, init='k-means++', random_state=0)
        model.fit(df_iris)
        # Building a list that holds lists with 1 feature the we can use scale later
        inertia_list.append(model.inertia_)
    save_plot(inertia_list)


# The function received a list of inertia
# The function saves a plot with a circle on the Elbow point and arrow aimed at it
def save_plot(inertia_list):
    matplotlib.use('Agg')  # For saving the figure as png
    plt.plot(range(1, MAX_K + 1), inertia_list, 'b-')
    plt.plot(3, inertia_list[2], 'o', ms=20, mec='k', mfc='none')  # Adding circle around the elbow point
    arrow_style = dict(arrowstyle='->', linestyle='--')  # Adding arrow to the elbow point
    plt.annotate('Elbow Point', xy=(3, inertia_list[2]), arrowprops=arrow_style, xytext=(4, 200))
    plt.xticks(range(1, MAX_K + 1))
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.title('Elbow Method for selection of optimal "K" clusters')
    plt.savefig(FIG_FILE_PIC)


if __name__ == '__main__':
    main()
