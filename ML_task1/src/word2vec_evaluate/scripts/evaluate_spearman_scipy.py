import gensim
import pandas as pd
from scipy.stats import spearmanr
# 不需要导入 nltk.metrics.spearman
# from nltk.metrics import spearman
from scipy.stats import rankdata

# 加载预训练的 Word2Vec 模型
model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)

word_sim_data = pd.read_csv('../data/wordsim353.tsv', sep='\t', comment='#')

word_sim_pairs = list(zip(word_sim_data['Word1'], word_sim_data['Word2'], word_sim_data['Human(mean)']))
# for i in range(len(word_sim_pairs)):
#     print(word_sim_pairs[i][0], word_sim_pairs[i][1],word_sim_pairs[i][2])

predicted_similarities = []
true_similarities = []

for word1, word2, true_sim in word_sim_pairs:
    try:
        # 获取词向量
        vec1 = model[word1]
        vec2 = model[word2]
        # 计算余弦相似度
        sim = model.cosine_similarities(vec1, [vec2])[0]

        predicted_similarities.append(float(sim))
        true_similarities.append(float(true_sim))
    except KeyError:
        # 如果模型中没有某个词，就跳过
        continue

# 打印数据以检查格式
print(true_similarities[:10])  # 打印前10个真实相似度值
print(predicted_similarities[:10])  # 打印前10个预测相似度值

# 计算 Spearman 相关系数
spearman_corr, p_value = spearmanr(true_similarities, predicted_similarities)
print(f"Spearman's correlation coefficient: {spearman_corr}")
print(f"P-value: {p_value}")
