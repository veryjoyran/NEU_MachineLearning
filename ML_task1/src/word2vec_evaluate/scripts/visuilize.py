import gensim
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_tsne(model, model_name, words_to_visualize):
    # 提取词向量
    word_vectors = []
    words = []

    for word in words_to_visualize:
        try:
            # 获取词向量
            vec = model[word]
            word_vectors.append(vec)
            words.append(word)
        except KeyError:
            continue

    word_vectors = np.array(word_vectors)

    # 使用 t-SNE 将词向量降维到 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_vectors = tsne.fit_transform(word_vectors)

    # 可视化词向量（2D）
    plt.figure(figsize=(12, 10))
    plt.xlim()
    plt.ylim()
    for i, word in enumerate(words):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
        plt.text(reduced_vectors[i, 0] + 0.05, reduced_vectors[i, 1] + 0.05, word, fontsize=9)

    plt.title(f"2D t-SNE visualization of Word2Vec embeddings ({model_name})")
    plt.show()

    # 使用 t-SNE 将词向量降维到 3D
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
    reduced_vectors_3d = tsne_3d.fit_transform(word_vectors)

    # 可视化词向量（3D）
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, word in enumerate(words):
        ax.scatter(reduced_vectors_3d[i, 0], reduced_vectors_3d[i, 1], reduced_vectors_3d[i, 2])
        ax.text(reduced_vectors_3d[i, 0] + 0.05, reduced_vectors_3d[i, 1] + 0.05, reduced_vectors_3d[i, 2] + 0.05, word, fontsize=9)

    plt.title(f"3D t-SNE visualization of Word2Vec embeddings ({model_name})")
    plt.show()

# 加载预训练的 Word2Vec 模型（不同维度）
model_50d = gensim.models.KeyedVectors.load_word2vec_format('../data/glove.6B/glove.6B.50d.w2v', binary=False)
model_100d = gensim.models.KeyedVectors.load_word2vec_format('../data/glove.6B/glove.6B.100d.w2v', binary=False)
model_200d = gensim.models.KeyedVectors.load_word2vec_format('../data/glove.6B/glove.6B.200d.w2v', binary=False)
model_300d = gensim.models.KeyedVectors.load_word2vec_format('../data/glove.6B/glove.6B.300d.w2v', binary=False)

# 读取 WordSim353 数据集
word_sim_data = pd.read_csv('../data/wordsim353.tsv', sep='\t', comment='#')

# 提取所有的词
all_words = list(set(word_sim_data['Word1']).union(set(word_sim_data['Word2'])))

# 选择前100个词进行可视化，可以调整这个值
words_to_visualize = all_words[:100]

# words_to_visualize = ['king', 'queen', 'man', 'woman', 'paris', 'france', 'computer', 'keyboard', 'dog', 'cat']


# 可视化不同维度模型的 t-SNE
for model, model_name in zip([model_50d, model_100d, model_200d, model_300d], ['50d', '100d', '200d', '300d']):
    visualize_tsne(model, model_name, words_to_visualize)
