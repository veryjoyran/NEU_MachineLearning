import json
import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import dok_matrix, load_npz


# 加载新闻数据
def load_news_data(json_file):
    """
    加载新闻数据，读取多个 JSON 对象
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            try:
                obj = json.loads(line)  # 使用 json.loads() 逐行读取并解析 JSON 对象
                data.append(obj)
            except json.JSONDecodeError:
                continue  # 如果某行不是有效的 JSON 格式，则跳过
    return data


# 加载停用词
def load_stopwords(stopwords_file):
    """
    加载停用词列表
    """
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = json.load(f)
    return stopwords


# 文本预处理：分词并去除停用词
def preprocess_text(text, stopwords):
    """
    分词并去除停用词
    """
    # 由于传入的 text 是字典类型，需要从中提取出 'content' 字段
    if isinstance(text, dict):  # 如果 text 是字典，提取 'content'
        text = text.get('content', '')

    # 使用 jieba 进行分词
    words = list(jieba.cut(text))  # 将生成器转换为列表
    # print("原始词语:", words[:10])  # 打印前10个词（调试用）

    # 去除停用词并过滤掉空格
    filtered_words = [word for word in words if word not in stopwords and word.strip() != ""]
    # print("去停用词后的词语:", filtered_words[:10])  # 打印去停用词后的前10个词（调试用）

    return " ".join(filtered_words)


# 构建 Term-Term 共现矩阵（使用稀疏矩阵）
def build_term_term_matrix(corpus, window_size=5, min_df=2, save_path="./data/co_occurrence_matrix.npz", feature_names_path="./data/feature_names.json"):
    # 如果已经存在文件，就直接加载
    try:
        co_occurrence_matrix = load_npz(save_path)
        with open(feature_names_path, 'r', encoding='utf-8') as f:
            feature_names = json.load(f)
        print("已加载预先保存的共现矩阵和词汇表。")
    except FileNotFoundError:
        vectorizer = CountVectorizer(analyzer='word', stop_words='english', min_df=min_df)
        X = vectorizer.fit_transform(corpus)

        feature_names = vectorizer.get_feature_names_out()
        vocab_size = len(feature_names)

        co_occurrence_matrix = dok_matrix((vocab_size, vocab_size), dtype=np.float64)

        for i, doc in enumerate(corpus):
            words = doc.split()
            for j, word in enumerate(words):
                if word not in feature_names:
                    continue
                word_index = feature_names.tolist().index(word)
                start = max(0, j - window_size)
                end = min(len(words), j + window_size + 1)
                for k in range(start, end):
                    if k != j:
                        other_word = words[k]
                        if other_word in feature_names:
                            other_word_index = feature_names.tolist().index(other_word)
                            co_occurrence_matrix[word_index, other_word_index] += 1

        # 保存共现矩阵和词汇表
        save_npz(save_path, co_occurrence_matrix)
        with open(feature_names_path, 'w', encoding='utf-8') as f:
            json.dump(feature_names.tolist(), f)
        print("共现矩阵和词汇表已保存。")

    return co_occurrence_matrix, feature_names

# 主程序
def main():
    news_file = './data/news2016zh_valid_2000.json'  # 新闻数据文件路径
    stopwords_file = './data/stopwords-zh.json'  # 停用词文件路径

    corpus = load_news_data(news_file)
    print("加载的新闻数据示例：", corpus[:2])

    stopwords = load_stopwords(stopwords_file)

    # 数据预处理：分词和去除停用词
    processed_corpus = [preprocess_text(article, stopwords) for article in corpus]

    # 构建 Term-Term 矩阵（共现矩阵）
    co_occurrence_matrix, feature_names = build_term_term_matrix(processed_corpus, window_size=5)

    # 查看共现矩阵的前五个词汇的共现情况
    print("Co-occurrence matrix for first 5 words:")
    print(co_occurrence_matrix[:5, :5])

if __name__ == "__main__":
    main()