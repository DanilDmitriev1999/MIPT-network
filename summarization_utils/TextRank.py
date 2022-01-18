from itertools import combinations
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.linalg import norm
import razdel

class TextRank:
    def __init__(self, model_name:str):
        self.encoder = SentenceTransformer(model_name)

    def __model_similarity(self, hash_vec, idx_1, idx_2):
        u = hash_vec[idx_1]
        v = hash_vec[idx_2]

        return self.__cosine_sim(u, v)

    def __gen_text_rank_summary(self, text, lower=True):
        # Разбиваем текст на предложения
        sentences = [sentence.text for sentence in razdel.sentenize(text)]
        n_sentences = len(sentences)
        n_sentence = 2 if n_sentences < 15 else 5

        # Токенизируем предложения
        sentences_words = [[token.text.lower() if lower else token.text for token in razdel.tokenize(sentence)] for
                           sentence in sentences]

        # хешируем предложения и их вектора
        hash_vec = {idx:np.mean(self.encoder.encode([' '.join(i)]), axis=0) for idx, i in enumerate(sentences_words)}

        # Для каждой пары предложений считаем близость
        pairs = combinations(range(n_sentences), 2)
        scores = [(i, j, self.__model_similarity(hash_vec, i, j)) for i, j in pairs]

        # Строим граф с рёбрами, равными близости между предложениями
        g = nx.Graph()
        g.add_weighted_edges_from(scores)

        # Считаем PageRank
        pr = nx.pagerank(g, max_iter=1500)
        result = [(i, pr[i], s) for i, s in enumerate(sentences) if i in pr]
        result.sort(key=lambda x: x[1], reverse=True)

        # Выбираем топ предложений
        result = result[:n_sentence]

        # Восстанавливаем оригинальный их порядок
        result.sort(key=lambda x: x[0])

        # Восстанавливаем текст выжимки
        predicted_summary = " ".join([sentence for i, proba, sentence in result])
        predicted_summary = predicted_summary.lower() if lower else predicted_summary
        return predicted_summary

    def get_summary(self, records, lower=True):
        predicted_summary = self.__gen_text_rank_summary(records, lower)
        return predicted_summary

    @staticmethod
    def __cosine_sim(u, v):
        return np.dot(u, v) / (norm(u) * norm(v))