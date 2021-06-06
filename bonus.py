import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


MAX_K = 10
FIG_FILE_PIC = "elbow.png"


def main():
    ds_iris = datasets.load_iris()
    df_iris = ds_iris.data
    inertia_list = []
    for k in range(1, MAX_K + 1):
        model = KMeans(n_clusters=k, init='k-means++', random_state=0)
        model.fit(df_iris)
        inertia_list.append([model.inertia_])
    save_plot(inertia_list)


def save_plot(inertia_list):
    scale = MinMaxScaler()
    scaled = scale.fit_transform(inertia_list)
    plt.plot(range(1, MAX_K + 1), scaled, 'b-')
    plt.plot(3, scaled[2], 'o', ms=20, mec='k', mfc='none')
    plt.annotate('Elbow Point', xy=(3, scaled[2]), arrowprops=dict(arrowstyle='->', linestyle='--'), xytext=(4, 0.6))
    plt.xticks(range(1, MAX_K + 1))
    plt.xlabel('K')
    plt.ylabel('Normalized inertia')
    plt.title('Elbow Method for selection of optimal "K" clusters')
    plt.savefig(FIG_FILE_PIC)


if __name__ == '__main__':
    main()
