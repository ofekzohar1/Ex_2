import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


MAX_K = 10
FIG_FILE_PIC = "elbow.png" #file saved name


# Using Kmeans ++ algoritem with random state 0 is needed on the data from iris 
# Using save_plot function
def main():
    ds_iris = datasets.load_iris()
    df_iris = ds_iris.data
    inertia_list = []
    for k in range(1, MAX_K + 1):
        model = KMeans(n_clusters=k, init='k-means++', random_state=0)
        model.fit(df_iris)
        inertia_list.append([model.inertia_]) #building a list that holdes lists with 1 feature the we can use scale later
    save_plot(inertia_list)


# The function recived a lisr of inertia, scales it between 0-1
# The function saves a plot with a circle on the Elbow point and arrow aimed at it
def save_plot(inertia_list):
    scale = MinMaxScaler()
    scaled = scale.fit_transform(inertia_list)
    plt.plot(range(1, MAX_K + 1), scaled, 'b-')
    plt.plot(3, scaled[2], 'o', ms=20, mec='k', mfc='none') #adding circle around the elbow point
    plt.annotate('Elbow Point', xy=(3, scaled[2]), arrowprops=dict(arrowstyle='->', linestyle='--'), xytext=(4, 0.6)) #adding arrow to the elbow poinr
    plt.xticks(range(1, MAX_K + 1))
    plt.xlabel('K')
    plt.ylabel('Normalized inertia')
    plt.title('Elbow Method for selection of optimal "K" clusters')
    plt.savefig(FIG_FILE_PIC)


if __name__ == '__main__':
    main()
