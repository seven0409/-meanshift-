from pywebio.input import file_upload
from pywebio.output import put_text,put_html,put_image,put_table
from pywebio import start_server
from pyecharts.charts import Bar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from itertools import cycle
from sklearn.cluster import MeanShift, estimate_bandwidth
from PIL import Image

def meanshift():

        upload_file=file_upload(label="data")
        filepath='./updata_file.csv'
        with open(filepath,'wb')as file:
            file.write(upload_file['content'])
        data = pd.read_csv(filepath)
        data1 = data["START_LNG"]  # 开始经度
        data2 = data["START_LAT"]  # 开始纬度
        data3 = data["END_LNG"]  # 结束经度
        data4 = data["END_LAT"]  # 结束纬度
        cluster1 = [data2, data1]
        cluster2 = [data4, data3]
        X = np.hstack((cluster1, cluster2)).T
        bandwidth = estimate_bandwidth(X, quantile=0.3)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        # 聚合点的经纬度坐标
        put_text(cluster_centers)
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        # 相当于kmean中k值
        put_text("number of estimated clusters : %d" % n_clusters_)
        plt.figure(1)
        plt.clf()

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        # 聚类散点图
        plt.savefig("D:/scatter.jpg")
        img_cv = Image.open("D:/scatter.jpg")
        put_image(img_cv)
      #  plt.show()
        # 轮廓系数：聚类结果的轮廓系数的取值在[-1,1]之间，值越大，说明同类样本相距约近，不同样本相距越远，则聚类效果越好。
        put_text(metrics.silhouette_score(X, labels, metric='euclidean', sample_size=None, random_state=None))

if __name__ == '__main__':

   start_server(meanshift,8080)