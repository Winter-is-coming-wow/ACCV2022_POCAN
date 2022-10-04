# -- coding: utf-8 --
import tensorflow as tf
import numpy as np
#from gl import Config
#from Dataloader import Fer2013dataset,RAFdataset,Fer2013Plus,DataLoader
# import Bio.Cluster as cluster
# import numpy as np
# from munkres import Munkres
# from sklearn.metrics.pairwise import cosine_similarity

# def run_kmeans(target,source_proto,num_cluster=7):
#     source_proto=np.squeeze(source_proto)
#     logits = cosine_similarity(target, source_proto)
#     target_pseudo_label = np.argmax(logits, axis=-1)
#     target_clusterid,error,nfound=cluster.kcluster(target,nclusters=num_cluster,npass=3,dist='u',initialid=target_pseudo_label)
#     target_proto,cmask=cluster.clustercentroids(target,clusterid=target_clusterid)
#     target_proto=target_proto.astype(np.float32)
    
#     return target_proto
from faiss import Kmeans as faiss_Kmeans



def run_kmeans(target,source_proto,num_cluster=7):
    # print(source_proto.shape)
    # print(target.shape)
    source_proto=np.squeeze(source_proto)
    kmeans = faiss_Kmeans(
            target.shape[-1],
            num_cluster,
            niter=20,
            verbose=False,
            nredo = 3,
            spherical=True,
            min_points_per_centroid=1,
            max_points_per_centroid=10000,
            gpu=True,
            frozen_centroids=False,
        )

    kmeans.train(target, init_centroids=source_proto)

    # _, I = kmeans.index.search(data, 1)
    # labels.append(I.squeeze(1))
    C = kmeans.centroids
    #centroids.append(C)

    #labels = np.stack(labels, axis=0)

    return C