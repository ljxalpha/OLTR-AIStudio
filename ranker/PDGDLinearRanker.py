# -*- coding: utf-8 -*-
from ranker.LinearRanker import LinearRanker
import numpy as np

class PDGDLinearRanker(LinearRanker):
    def __init__(self, num_features, learning_rate, tau, learning_rate_decay=1, random_initial=True):
        # py3
        # super().__init__(num_features, learning_rate, learning_rate_decay, random_initial)
        # py2
        LinearRanker.__init__(self, num_features, learning_rate, learning_rate_decay, random_initial)
        self.tau = tau
    # 返回与当前qid相关的所有文档的分数，以及采样出的k个文档构成的结果列表
    def get_query_result_list(self, dataset, query):
        # 获取到当前qid所关联的所有文档的QD特征向量构成的矩阵
        feature_matrix = dataset.get_all_features_by_query(query)
        # 获取到当前qid所关联的额所有文档的从0开始编码的docid
        docid_list = np.array(dataset.get_candidate_docids_by_query(query))
        n_docs = docid_list.shape[0] #统计与qid相关联的文档数
        # 给用户展示不超过10个文档
        k = np.minimum(10, n_docs)
        # 这个分数就是所有query_doc的特征矩阵乘以线性模型的权重向量得到的
        doc_scores = self.get_scores(feature_matrix)
        # 通过这样一个操作之后,doc_scores的最高分就变成18？？？？？？？？？？？？？？？
        doc_scores += 18 - np.amax(doc_scores)
        # 这是给出打分后的列表的过程，怎么实现的还没看懂，需要看PDGD的原理和公式
        ranking = self._recursive_choice(np.copy(doc_scores),
                                         np.array([], dtype=np.int32),
                                         k)
        return ranking, doc_scores
    #这个函数是使用论文的公式1来对集合中对的文档进行采用，从而得到结果列表
    def _recursive_choice(self, scores, incomplete_ranking, k_left):
        n_docs = scores.shape[0]
        # 每一轮都将上一轮已经添加到结果列表中的文档的概率最小化，以防止对已添加到结果列表中的文档重复添加
        scores[incomplete_ranking] = np.amin(scores)

        scores += 18 - np.amax(scores)
        exp_scores = np.exp(scores/self.tau)
        # 已经添加到结果列表的文档对应的指数分数要置0
        exp_scores[incomplete_ranking] = 0
        probs = exp_scores / np.sum(exp_scores)# probs 是公式1求出的每一个文档在当前文档集合上的概率分布
        # 找出文档概率大于10e-4/文档数的文档的数量
        safe_n = np.sum(probs > 10 ** (-4) / n_docs)

        safe_k = np.minimum(safe_n, k_left)
        # 从所有的文档中，依据文档的概率probs选择出safek个文档组成结果列表，replace=False表示采样的结果不允许重复元素
        next_ranking = np.random.choice(np.arange(n_docs),
                                        replace=False,
                                        p=probs,
                                        size=safe_k)
        # 每一轮都选择safek个文档添加到结果列表中，下一轮继续添加，知道列表中的总数等于最初的k
        ranking = np.concatenate((incomplete_ranking, next_ranking))
        k_left = k_left - safe_k

        if k_left > 0:
            return self._recursive_choice(scores, ranking, k_left)
        else:
            return ranking

    def update_to_clicks(self, click_label, ranking, doc_scores, feature_matrix, last_exam=None):

        if last_exam is None:
            # 将01的click_label转成TRUE、FALSE向量
            clicks = np.array(click_label == 1)
            # 结果列表的文档数
            n_docs = ranking.shape[0]
            n_results = 10
            cur_k = np.minimum(n_docs, n_results)

            included = np.ones(cur_k, dtype=np.int32)
            # 如果最后一个文档没有被点击
            if not clicks[-1]:
                included[1:] = np.cumsum(clicks[::-1])[:0:-1]#将clicks逆序，然后逐位累加求和，然后将截取第二个到最后一个元素，再逆序；其意义可以理解为“截止到当前位置，还剩多少个已经发生的点击（除了最后一个文档）”
            # pos_ind是发生点击的位置（已点击的位置），neg_ind是发生点击之前和后一个的位置（用户已检查的位置）；这是基于PDGD的用户检查行为假设
            neg_ind = np.where(np.logical_xor(clicks, included))[0]
            pos_ind = np.where(clicks)[0]

        else:

            if last_exam == 10:
                neg_ind = np.where(click_label[:last_exam] == 0)[0]
                pos_ind = np.where(click_label[:last_exam] == 1)[0]
            else:
                neg_ind = np.where(click_label[:last_exam + 1] == 0)[0]
                pos_ind = np.where(click_label[:last_exam] == 1)[0]

        # 下面要构造pairwise
        n_pos = pos_ind.shape[0]
        n_neg = neg_ind.shape[0]
        n_pairs = n_pos * n_neg # 一共的pair数

        if n_pairs == 0:
            return
        #点击文档和检查文档的docid
        pos_r_ind = ranking[pos_ind]
        neg_r_ind = ranking[neg_ind]
        # 点击文档和检查文档的文档分数
        pos_scores = doc_scores[pos_r_ind]
        neg_scores = doc_scores[neg_r_ind]
        # 将分数复制，注意这里调用了两个不同的复制方法，tile和repeat
        log_pair_pos = np.tile(pos_scores, n_neg)
        log_pair_neg = np.repeat(neg_scores, n_pos)
        # np.maximum会逐位比较两个数组，将每一位的最大值构造成一个新的数组
        pair_trans = 18 - np.maximum(log_pair_pos, log_pair_neg)
        exp_pair_pos = np.exp(log_pair_pos + pair_trans)
        exp_pair_neg = np.exp(log_pair_neg + pair_trans)
        # 这一项是算法第10行的分母
        pair_denom = (exp_pair_pos + exp_pair_neg)
        pair_w = np.maximum(exp_pair_pos, exp_pair_neg)
        pair_w /= pair_denom #第10行分母，除以平方项
        pair_w /= pair_denom
        pair_w *= np.minimum(exp_pair_pos, exp_pair_neg) # 111行和107行完成算法第10行分子的计算
        # 第9行，ρ函数的计算
        pair_w *= self._calculate_unbias_weights(pos_ind, neg_ind, doc_scores, ranking)
        # reshaped的每一行表示一个检查文档依次和所有点击文档构成的文档对经由算法的9行和10行的计算结果
        reshaped = np.reshape(pair_w, (n_neg, n_pos))
        pos_w = np.sum(reshaped, axis=0) # 一行，每一列表示一个点击文档依次和所有点击文档构成的文档对的计算结果的和（若行以数字编号，列以字母编号，则第一个格为A1+A2+A3;第二个格为B1+B2+B3）
        neg_w = -np.sum(reshaped, axis=1) # 一行，每一列表示一个检查文档依次和所有点击文档构成的文档对的计算结果的和（若行以数字编号，列以字母编号，则第一个格为A1+B1+C1；第二个格为A2+B2+C2）

        all_w = np.concatenate([pos_w, neg_w])
        all_ind = np.concatenate([pos_r_ind, neg_r_ind])
        # 算法第11行和12行的计算
        self._update_to_documents(all_ind, all_w, feature_matrix)

    def _update_to_documents(self, doc_ind, doc_weights, feature_matrix):
        weighted_docs = feature_matrix[doc_ind, :] * doc_weights[:, None]
        gradient = np.sum(weighted_docs, axis=0)
        # print("gradient length", np.sqrt(np.sum(gradient ** 2)))
        # # print(gradient)
        # print("weight length", np.sqrt(np.sum(self.weights**2)))
        # # print(self.weights)
        # print()
        self.weights += self.learning_rate * gradient
        self.learning_rate *= self.learning_rate_decay

    def _calculate_unbias_weights(self, pos_ind, neg_ind, doc_scores, ranking):
        ranking_prob = self._calculate_observed_prob(pos_ind, neg_ind,
                                                     doc_scores, ranking) # 没有反转的排序的概率
        flipped_prob = self._calculate_flipped_prob(pos_ind, neg_ind,
                                                    doc_scores, ranking) # 反转后的排序的概率
        return flipped_prob / (ranking_prob + flipped_prob)

    def _calculate_flipped_prob(self, pos_ind, neg_ind, doc_scores, ranking):
        n_pos = pos_ind.shape[0]
        n_neg = neg_ind.shape[0]
        n_pairs = n_pos * n_neg
        n_results = ranking.shape[0]
        n_docs = doc_scores.shape[0]

        results_i = np.arange(n_results)
        pair_i = np.arange(n_pairs)
        doc_i = np.arange(n_docs)

        pos_pair_i = np.tile(pos_ind, n_neg)
        neg_pair_i = np.repeat(neg_ind, n_pos)

        flipped_rankings = np.tile(ranking[None, :],
                                   [n_pairs, 1])
        flipped_rankings[pair_i, pos_pair_i] = ranking[neg_pair_i]
        flipped_rankings[pair_i, neg_pair_i] = ranking[pos_pair_i]

        min_pair_i = np.minimum(pos_pair_i, neg_pair_i)
        max_pair_i = np.maximum(pos_pair_i, neg_pair_i)
        range_mask = np.logical_and(min_pair_i[:, None] <= results_i,
                                    max_pair_i[:, None] >= results_i)

        flipped_log = doc_scores[flipped_rankings]

        safe_log = np.tile(doc_scores[None, None, :],
                           [n_pairs, n_results, 1])

        results_ij = np.tile(results_i[None, 1:], [n_pairs, 1])
        pair_ij = np.tile(pair_i[:, None], [1, n_results - 1])
        mask = np.zeros((n_pairs, n_results, n_docs))
        mask[pair_ij, results_ij, flipped_rankings[:, :-1]] = True
        mask = np.cumsum(mask, axis=1).astype(bool)

        safe_log[mask] = np.amin(safe_log)
        safe_max = np.amax(safe_log, axis=2)
        safe_log -= safe_max[:, :, None] - 18
        flipped_log -= safe_max - 18
        flipped_exp = np.exp(flipped_log)

        safe_exp = np.exp(safe_log)
        safe_exp[mask] = 0
        safe_denom = np.sum(safe_exp, axis=2)
        safe_prob = np.ones((n_pairs, n_results))
        safe_prob[range_mask] = (flipped_exp / safe_denom)[range_mask]

        safe_pair_prob = np.prod(safe_prob, axis=1)

        return safe_pair_prob

    def _calculate_observed_prob(self, pos_ind, neg_ind, doc_scores, ranking):
        n_pos = pos_ind.shape[0]
        n_neg = neg_ind.shape[0]
        n_pairs = n_pos * n_neg
        n_results = ranking.shape[0]
        n_docs = doc_scores.shape[0]

        results_i = np.arange(n_results)
        # pair_i = np.arange(n_pairs)
        # doc_i = np.arange(n_docs)

        pos_pair_i = np.tile(pos_ind, n_neg)
        neg_pair_i = np.repeat(neg_ind, n_pos)

        min_pair_i = np.minimum(pos_pair_i, neg_pair_i)
        max_pair_i = np.maximum(pos_pair_i, neg_pair_i)
        range_mask = np.logical_and(min_pair_i[:, None] <= results_i,
                                    max_pair_i[:, None] >= results_i)

        safe_log = np.tile(doc_scores[None, :],
                           [n_results, 1])

        mask = np.zeros((n_results, n_docs))
        mask[results_i[1:], ranking[:-1]] = True
        mask = np.cumsum(mask, axis=0).astype(bool)

        safe_log[mask] = np.amin(safe_log)
        safe_max = np.amax(safe_log, axis=1)
        safe_log -= safe_max[:, None] - 18
        safe_exp = np.exp(safe_log)
        safe_exp[mask] = 0

        ranking_log = doc_scores[ranking] - safe_max + 18
        ranking_exp = np.exp(ranking_log)

        safe_denom = np.sum(safe_exp, axis=1)
        ranking_prob = ranking_exp / safe_denom

        tiled_prob = np.tile(ranking_prob[None, :], [n_pairs, 1])

        safe_prob = np.ones((n_pairs, n_results))
        safe_prob[range_mask] = tiled_prob[range_mask]

        safe_pair_prob = np.prod(safe_prob, axis=1)

        return safe_pair_prob

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_tau(self, tau):
        self.tau = tau