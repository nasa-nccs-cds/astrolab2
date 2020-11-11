#%%
import cudf
import cuml
import pandas as pd
import datashader as ds
import datashader.utils as utils
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
pal = [
    "#9e0142",
    "#d8434e",
    "#f67a49",
    "#fdbf6f",
    "#feeda1",
    "#f1f9a9",
    "#bfe5a0",
    "#74c7a5",
    "#378ebb",
    "#5e4fa2",
]
color_key = {str(d): c for d, c in enumerate(pal)}

# Using cudf Dataframe here is not likely to help with performance
# However, it's a good opportunity to get familiar with the API
source_df: cudf.DataFrame = cudf.read_csv('/att/nobackup/tpmaxwel/data/fashion-mnist-csv/fashion_train.csv')
data = source_df.loc[ :, source_df.columns[:-1] ]
target = source_df[ source_df.columns[-1] ]

# # Compute K-NN graph
#
# import cudf
# from cuml.neighbors import NearestNeighbors
# from cuml.datasets import make_blobs
#
# X, _ = make_blobs( n_samples=25, centers=5, n_features=10, random_state=42 )
#
# # build a cudf Dataframe
# X_cudf = cudf.DataFrame(X)
#
# # fit model
# model = NearestNeighbors(n_neighbors=3)
# model.fit(X)
#
# # get 3 nearest neighbors
# distances, indices = model.kneighbors(X_cudf)
#
# # Need sparse array format.

reducer = cuml.UMAP(
    n_neighbors=15,
    n_components=3,
    n_epochs=500,
    min_dist=0.1,
    output_type="numpy"
)
embedding = reducer.fit_transform(data)
print(f"Completed embedding, shape = {embedding.shape}")

# df = embedding.to_pandas()
# df.columns = ["x", "y"]
# df['class'] = pd.Series([str(x) for x in target.to_array()], dtype="category")
#
# cvs = ds.Canvas(plot_width=400, plot_height=400)
# agg = cvs.points(df, 'x', 'y', ds.count_cat('class'))
# img = tf.shade(agg, color_key=color_key, how='eq_hist')
#
# utils.export_image(img, filename='fashion-mnist', background='black')
#
# image = plt.imread('fashion-mnist.png')
# fig, ax = plt.subplots(figsize=(12, 12))
# plt.imshow(image)
# plt.setp(ax, xticks=[], yticks=[])
# plt.title("Fashion MNIST data embedded\n"
#           "into two dimensions by UMAP\n"
#           "visualised with Datashader",
#           fontsize=12)
#
# plt.show()
#





