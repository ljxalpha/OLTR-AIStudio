# -*- coding: utf-8 -*-
class AbstractDataset:

    def __init__(self,
                 path,
                 feature_size,
                 query_level_norm=False):
        self._path = path
        self._feature_size = feature_size
        self._query_docid_get_features = {}
        # 每一个query的所有关联文档进行从0开始编号（无论相关系数是多少），注意这个docid是自己编号一个一个数出来的，不是从数据集里读出来的
        # 字典类型，key为qid，value为一个list，从0开始
        self._query_get_docids = {}
        # 每一个query的所有doc的特征的矩阵，每行对应一个query-doc对，行的数量与self._query_get_docids里相同qid所对应的列表长度相同
        # 字典类型，key为qid，value为二维的np.ndarray
        self._query_get_all_features = {}
        # 每一个query-doc对的相关分数标签
        # 类型为二级字典，一级key为qid，二级key为与self._query_get_docids相对应的docid
        self._query_docid_get_rel = {}
        # 每个query的相关的（relevance>0）的文档的docid（与self._query_get_docids相对应），按照在数据集的出现顺序排列，而不是按照分数高低排列
        # 字典类型，key为qid，value为所有relevance大于0的doc的id的列表
        self._query_pos_docids = {}
        # ？？？
        self._query_level_norm = query_level_norm

    def _load_data(self):
        raise NotImplementedError("Derived class needs to implement "
                                  "_load_data.")

    def get_features_by_query_and_docid(self, query, docid):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_features_by_query_and_docid.")

    def get_candidate_docids_by_query(self, query):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_candidate_docids_by_query.")

    def get_all_features_by_query(self, query):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_all_features_by_query.")

    def get_relevance_label_by_query_and_docid(self, query, docid):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_relevance_by_query_and_docid.")


    def get_relevance_docids_by_query(self, query):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_relevance_docids_by_query.")

    def get_all_querys(self):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_all_querys.")
