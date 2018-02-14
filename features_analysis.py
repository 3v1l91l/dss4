from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from ggplot import ggplot, ggtitle, geom_point, aes

users_df = pd.read_pickle('users_df')
users_df = users_df[~users_df['preferences'].isnull()]
data = np.stack(users_df['preferences'].values)

pca = PCA(n_components=3)
pca_result = pca.fit_transform(data)

pca_df = pd.DataFrame(data={'pca_1': pca_result[:,0], 'pca_2': pca_result[:,1], 'pca_3': pca_result[:,2]})

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

chart = ggplot( pca_df, aes(x='pca_1', y='pca_2') ) \
        + geom_point(size=75,alpha=0.8) \
        + ggtitle("First and Second Principal Components")
print(chart)
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
#
# tsne_results = tsne.fit_transform(data)
# df_tsne = pd.DataFrame(data={'x_tsne': tsne_results[:,0], 'y_tsne': tsne_results[:,1]})
# chart = ggplot( df_tsne, aes(x='x_tsne', y='y_tsne') ) \
#         + geom_point(size=70,alpha=0.1) \
#         + ggtitle("tSNE dimensions colored by digit")
# print(chart)
