import gensim
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from nltk.metrics import spearman

def calculate_spearman(model, word_sim_path):
    # 读取 WordSim353 数据集
    try:
        word_sim_data = pd.read_csv(word_sim_path, sep='\t', comment='#')
    except FileNotFoundError:
        print(f"文件未找到: {word_sim_path}")
        return None

    # 确保数据集包含必要的列
    required_columns = {'Word1', 'Word2', 'Human(mean)'}
    if not required_columns.issubset(word_sim_data.columns):
        print(f"数据集缺少必要的列: {required_columns}")
        return None

    word_sim_pairs = list(zip(word_sim_data['Word1'], word_sim_data['Word2'], word_sim_data['Human(mean)']))

    predicted_similarities = []
    true_similarities = []
    items = []

    for word1, word2, true_sim in word_sim_pairs:
        try:
            # 计算词对之间的相似度
            sim = model.similarity(word1, word2)
            predicted_similarities.append(float(sim))
            true_similarities.append(float(true_sim))
            items.append(f"{word1}-{word2}")
        except KeyError as e:
            # 如果模型中没有某个词，就跳过，并打印警告
            print(f"词汇未找到于模型中: {e}")
            continue

    if not predicted_similarities:
        print("没有有效的词对用于计算相关系数。")
        return None

    # 对相似度进行排名
    ranked_true_similarities = rankdata(true_similarities)
    ranked_predicted_similarities = rankdata(predicted_similarities)

    # 创建词对到排名的字典
    true_ranked_dict = {item: rank for item, rank in zip(items, ranked_true_similarities)}
    pred_ranked_dict = {item: rank for item, rank in zip(items, ranked_predicted_similarities)}

    # 计算 Spearman 相关系数
    spearman_corr = spearman.spearman_correlation(true_ranked_dict, pred_ranked_dict)
    print(f"Spearman's correlation coefficient: {spearman_corr}")
    return spearman_corr


# 加载预训练的 Word2Vec 模型（不同维度）
model_50d = gensim.models.KeyedVectors.load_word2vec_format('../data/glove.6B/glove.6B.50d.w2v', binary=False)
model_100d = gensim.models.KeyedVectors.load_word2vec_format('../data/glove.6B/glove.6B.100d.w2v', binary=False)
model_200d = gensim.models.KeyedVectors.load_word2vec_format('../data/glove.6B/glove.6B.200d.w2v', binary=False)
model_300d = gensim.models.KeyedVectors.load_word2vec_format('../data/glove.6B/glove.6B.300d.w2v', binary=False)
model_google=gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)

spearman_corr_google = calculate_spearman(model_google,'../data/wordsim353.tsv')
print(f"google_pearman's correlation coefficient:{spearman_corr_google}")

# 创建一个空的列表来存储模型名称和斯皮尔曼相关系数
results = []

# 调用计算斯皮尔曼相关系数的函数，传入不同的模型并输出结果
for model, name in zip([model_50d, model_100d, model_200d, model_300d], ['50d', '100d', '200d', '300d']):
    spearman_corr = calculate_spearman(model,'../data/wordsim353.tsv')
    results.append([name, spearman_corr])

# 将结果转换为 pandas DataFrame
df = pd.DataFrame(results, columns=["Model", "Spearman's Correlation"])

# 绘制折线图
plt.figure(figsize=(8, 6))
plt.plot(df["Model"], df["Spearman's Correlation"], marker='o', linestyle='-', color='b')

# 在每个点上显示斯皮尔曼相关系数的值
for i in range(len(df)):
    plt.text(df["Model"][i], df["Spearman's Correlation"][i],
             df["Spearman's Correlation"][i],
             ha='center', va='bottom', fontsize=9)


# 设置图形的标题和标签
plt.title("Spearman's Correlation Coefficient for Different Models(50,100,200,300d)")
plt.xlabel("Model Dimensions")
plt.ylabel("Spearman's Correlation Coefficient")

# 显示图形
plt.grid(True)
plt.show()
