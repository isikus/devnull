__author__ = 'alenush'

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import copy
import re
import numpy as np
import logging
from gensim.models.fasttext import load_facebook_vectors
import difflib
from collections import defaultdict
#from auto_correction import *

logging.basicConfig(format = u'%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s',
                    level = logging.DEBUG, filename = u'classifier_log.log')

#with open('dictionary.json', 'r', encoding='utf-8') as f:
#    dictionary = json.loads(f.read())
#    dictionary['и'] = 1
#    print("Loading")
#    CORRECTOR = ErrorCorrection('unigrams.pkl', 'bigrams.pkl', 'trigrams.pkl', dictionary)
#    print("Done")

def extract_words(line):
    sents = line.split()
    words = []
    for word in sents:
        word = word.lower().replace('ё', 'е')
        i = len(word) - 1
        while i >= 0 and not (word[i].isalpha() or word[i].isdigit()):
            i -= 1
        if i >= 0:
            words.append(word[:(i+1)])
    return words

def make_levenstein_table(source, correct, allow_transpositions=False,
        removal_cost=1.0, insertion_cost=1.0, replace_cost=1.0, transposition_cost=1.0):
    """
    Строит динамическую таблицу, применяемую при вычислении расстояния Левенштейна,
    а также массив обратных ссылок, применяемый при восстановлении выравнивания

    :param source: list of strs, исходное предложение
    :param correct: list of strs, исправленное предложение
    :param allow_transpositions: bool, optional(default=False),
        разрешены ли перестановки соседних символов в расстоянии Левенштейна
    :param removal_cost: float, optional(default=1.0),
        штраф за удаление
    :param insertion_cost: float, optional(default=1.0),
        штраф за вставку
    :param replace_cost: float, optional(default=1.9),
        штраф за замену символов
    :param transposition_cost: float, optional(default=1.9),
        штраф за перестановку символов
    :return:
        table, numpy 2D-array of float, двумерная таблица расстояний между префиксами,
            table[i][j] = d(source[:i], correct[:j])
        backtraces, 2D-array of lists,
            двумерный массив обратных ссылок при вычислении оптимального выравнивания
    """
    first_length, second_length = len(source), len(correct)
    table = np.zeros(shape=(first_length + 1, second_length + 1), dtype=float)
    backtraces = [([None]  * (second_length + 1)) for _ in range(first_length + 1)]
    for i in range(1, second_length + 1):
        table[0][i] = i
        backtraces[0][i] = [(0, i-1)]
    for i in range(1, first_length + 1):
        table[i][0] = i
        backtraces[i][0] = [(i-1, 0)]
    for i, first_word in enumerate(source, 1):
        for j, second_word in enumerate(correct, 1):
            if first_word == second_word:
                table[i][j] = table[i-1][j-1]
                backtraces[i][j] = [(i-1, j-1)]
            else:
                table[i][j] = min((table[i-1][j-1] + replace_cost,
                                   table[i][j-1] + removal_cost,
                                   table[i-1][j] + insertion_cost))
                if (allow_transpositions and min(i, j) >= 2
                        and first_word == correct[j-2] and second_word == source[j-2]):
                    table[i][j] = min(table[i][j], table[i-2][j-2] + transposition_cost)
                curr_backtraces = []
                if table[i-1][j-1] + replace_cost == table[i][j]:
                    curr_backtraces.append((i-1, j-1))
                if table[i][j-1] + removal_cost == table[i][j]:
                    curr_backtraces.append((i, j-1))
                if table[i-1][j] + insertion_cost == table[i][j]:
                    curr_backtraces.append((i-1, j))
                if (allow_transpositions and min(i, j) >= 2
                    and first_word == correct[j-2] and second_word == source[j-2]
                        and table[i][j] == table[i-2][j-2] + transposition_cost):
                    curr_backtraces.append((i-2, j-2))
                backtraces[i][j] = copy.copy(curr_backtraces)
    return table, backtraces

def extract_best_alignment(backtraces):
    """
    Извлекает оптимальное выравнивание из таблицы обратных ссылок

    :param backtraces, 2D-array of lists,
        двумерный массив обратных ссылок при вычислении оптимального выравнивания
    :return: best_paths, list of lists,
        список путей, ведущих из точки (0, 0) в точку (m, n) в массиве backtraces
    """
    m, n = len(backtraces) - 1, len(backtraces[0]) - 1
    used_vertexes = {(m, n)}
    reverse_path_graph = defaultdict(list)
    vertexes_queue = [(m, n)]
    # строим граф наилучших путей в таблице
    while len(vertexes_queue) > 0:
        i, j = vertex = vertexes_queue.pop(0)
        if i > 0 or j > 0:
            for new_vertex in backtraces[i][j]:
                reverse_path_graph[new_vertex].append(vertex)
                if new_vertex not in used_vertexes:
                    vertexes_queue.append(new_vertex)
                    used_vertexes.add(new_vertex)
    # проходим пути в обратном направлении
    best_paths = []
    current_path = [(0, 0)]
    last_indexes, neighbor_vertexes_list = [], []
    while len(current_path) > 0:
        if current_path[-1] != (m, n):
            children = reverse_path_graph[current_path[-1]]
            if len(children) > 0:
                current_path.append(children[0])
                last_indexes.append(0)
                neighbor_vertexes_list.append(children)
                continue
        else:
            best_paths.append(copy.copy(current_path))
        while len(last_indexes) > 0 and last_indexes[-1] == len(neighbor_vertexes_list[-1]) - 1:
            current_path.pop()
            last_indexes.pop()
            neighbor_vertexes_list.pop()
        if len(last_indexes) == 0:
            break
        last_indexes[-1] += 1
        current_path[-1] = neighbor_vertexes_list[-1][last_indexes[-1]]
    return best_paths

def extract_basic_alignment_paths(paths_in_alignments, source, correct):
    """
    Извлекает из путей в таблице Левенштейна тождественные замены в выравнивании

    :param paths_in_alignments: list of lists, список оптимальных путей
        в таблице из точки (0, 0) в точку (len(source), len(correct))
    :param source: str, исходная строка,
    :param correct: str, строка с исправлениями
    :return:
        answer: list, список вариантов тождественных замен в оптимальных путях
    """
    m, n = len(source), len(correct)
    are_symbols_equal = np.zeros(dtype=bool, shape=(m, n))
    for i, a in enumerate(source):
        for j, b in enumerate(correct):
            are_symbols_equal[i][j] = (a == b)
    answer = set()
    for path in paths_in_alignments:
        answer.add(tuple(elem for elem in path[1:]
                         if are_symbols_equal[elem[0]-1][elem[1]-1]))
    return list(answer)

def extract_levenstein_alignments(source, correct):
    """
    Находит позиции тождественных замен
    в оптимальном выравнивании между source и correct

    :param source: str. исходная строка
    :param correct: str, исправленная строка
    :return: basic_alignment_paths, list of lists of pairs of ints
        список позиций тождественных замен в оптимальном выравнивании
    """
    table, backtraces = make_levenstein_table(source, correct, replace_cost=1.9)
    paths_in_alignments = extract_best_alignment(backtraces)
    basic_alignment_paths = extract_basic_alignment_paths(paths_in_alignments, source, correct)
    return basic_alignment_paths

def get_partition_indexes(first, second):
    """
    Строит оптимальное разбиение на группы (ошибка, исправление)
    Группа заканчивается после first[i] и second[j], если пара из
    концов этих слов встречается в оптимальном пути в таблице Левенштейна
    для " ".join(first) и " ".join(second)

    :param first: list of strs, список исходных слов
    :param second: list of strs, их исправление
    :return: answer, list of pairs of ints,
        список пар (f[0], s[0]), (f[1], s[1]), ...
        отрезок second[s[i]: s[i+1]] является исправлением для first[f[i]: f[i+1]]
    """
    m, n = len(first), len(second)
    answer = [(0, 0)]
    if m <= 1 or n <= 1:
        answer += [(m, n)]
    elif m == 2 and n == 2:
        answer += [(1, 1), (2, 2)]
    else:
        levenstein_table, backtraces = make_levenstein_table(" ".join(first), " ".join(second))
        best_paths_in_table = extract_best_alignment(backtraces)
        good_partitions, other_partitions = set(), set()
        word_ends = [0], [0]
        last = -1
        for i, word in enumerate(first):
            last = last + len(word) + 1
            word_ends[0].append(last)
        last = -1
        for i, word in enumerate(second):
            last = last + len(word) + 1
            word_ends[1].append(last)
        for path in best_paths_in_table:
            current_indexes = [(0, 0)]
            first_pos, second_pos = 0, 0
            is_partition_good = True
            for i, j in path[1:]:
                if i > word_ends[0][first_pos]:
                    first_pos += 1
                if j > word_ends[1][second_pos]:
                    second_pos += 1
                if i == word_ends[0][first_pos] and j == word_ends[1][second_pos]:
                    if first_pos > current_indexes[-1][0] or second_pos > current_indexes[-1][1]:
                        current_indexes.append((first_pos, second_pos))
                        first_pos += 1
                        second_pos += 1
                    else:
                        is_partition_good = False
            if current_indexes[-1] == (m, n):
                if is_partition_good:
                    good_partitions.add(tuple(current_indexes))
                else:
                    other_partitions.add(tuple(current_indexes))
        if len(good_partitions) == 1:
            answer = list(good_partitions)[0]
        else:
            answer = list(other_partitions)[0]
    return answer

def align_sents(source, correct, return_only_different=False):
    """
    Возвращает индексы границ групп в оптимальном выравнивании

    :param source, correct: str, исходное и исправленное предложение
    :param return_only_different: следует ли возвращать только индексы нетождественных исправлений
    :return: answer, list of pairs of tuples,
        оптимальное разбиение на группы. Если answer[i] == ((i, j), (k, l)), то
        в одну группу входят source[i:j] и correct[k:l]
    """
    alignments = extract_levenstein_alignments(source, correct)
    m, n = len(source), len(correct)
    prev = 0, 0
    answer = []
    for i, j in alignments[0]:
        if i > prev[0] + 1 or j > prev[1] + 1:
            partition_indexes =\
                get_partition_indexes(source[prev[0]: i-1], correct[prev[1]: j-1])
            if partition_indexes is not None:
                for pos, (f, s) in enumerate(partition_indexes[:-1]):
                    answer.append(((prev[0] + f, prev[0] + partition_indexes[pos+1][0]),
                                   (prev[1] + s, prev[1] + partition_indexes[pos+1][1])))
            else:
                answer.append((prev[0], i-1), (prev[1], j-1))
        if not return_only_different:
            answer.append(((i-1, i), (j-1, j)))
        prev = i, j
    if m > prev[0] or n > prev[1]:
        partition_indexes =\
                get_partition_indexes(source[prev[0]: m], correct[prev[1]: n])
        for pos, (f, s) in enumerate(partition_indexes[:-1]):
                answer.append(((prev[0] + f, prev[0] + partition_indexes[pos+1][0]),
                               (prev[1] + s, prev[1] + partition_indexes[pos+1][1])))
    return answer

#======================================================================

import pickle
from math import log10

GOOD_DICTIONARY = set()

CONTEXT = []
TOKEN_LEN = []
BLACKLISTS = []
REPEATED = []
W2VCANDIDAT = []
#LM_EM = []

TRAIN_DICTIONARY = {'tok_len': TOKEN_LEN, 'blacklist': BLACKLISTS, 'repeats': REPEATED,
                    'word2vec': W2VCANDIDAT, 'context':CONTEXT}
TARGETS = []


ALL_RESULTS = []

MODEL = load_facebook_vectors("./cc.ru.300.bin")

class Word():

    def __init__(self, word):
        self.word = word
        self.token_length = len(word)


    def check_blacklist(self):
        """
        Check if the word in blacklist (number, latin or one-symbol)
        :return 0 if not in blacklist, 1 if in
        """
        pattern = '[0-9\\.\\:\\-\\/a-z]+'
        if len(self.word) == 1:
            return 1
        else:
            result = re.match(pattern, self.word)
            if result != None:
                print(self.word)
                return 1
            else:
                return 0


    def check_tripple_letters(self):
        """
        Check if the word has repated letters like оооочень
        :return 1 if has, 0 if not.
        """
        pattern_miss = '(.+)?([а-я])\\2{2,}(.+)?'
        result = re.match(pattern_miss, self.word)
        if result != None:
            return 1 #misspel
        else:
            return 0 #spell ok



    def check_word2vec(self):
        """
        Load modedl and find the candidate-words that looks
        more like original.
        :return: word2vec ratio with candidate
        """
        have_candidate = 0
        try:
            candidates_array = MODEL.most_similar(positive=[self.word], topn=10, restrict_vocab=50000) # var
            for candidate in candidates_array:
                if candidate[1] > 0.40:  #var
                    s = difflib.SequenceMatcher(None, candidate[0], self.word)
                    if s.ratio() > 0.65: #var
                        have_candidate = candidate[1]
                        break
        except:
            have_candidate = 0
        return have_candidate


class Bad_sentence(Word):

    def __init__(self, sentence):
        super().__init__(sentence)
        self.sentence = sentence
        self.context = []
        self.tokens_length = []
        self.words = []
        self.test_token_list = []
        self.test_blacklist = []
        self.test_repeated = []
        self.test_word2vec = []
        self.test_context = []
        #self.lm_em = []


    def find_sentence_info(self):
        for i in range(len(self.sentence)):
            one_word = Word(self.sentence[i])
            self.tokens_length.append(one_word.token_length)
            self.find_contex_length()
            self.words.append(one_word)
            have_candidate = one_word.check_word2vec() # similarity ratio
            blacklist = one_word.check_blacklist()
            repeats = one_word.check_tripple_letters()
            #model_prob = CORRECTOR.correct_word(one_word.word, self.sentence[:i], self.sentence[i+1:], result='prob')
            #LM_EM.append(model_prob)
            TOKEN_LEN.append(int(one_word.token_length))
            BLACKLISTS.append(blacklist)
            REPEATED.append(repeats)
            W2VCANDIDAT.append(have_candidate)
            CONTEXT.append(len(self.context))


    def find_sentence_info_test(self, model):
        for i in range(len(self.sentence)):
            #new_word = replace_bad_words(word)
            #print(word, new_word)
            one_word = Word(self.sentence[i])
            self.tokens_length.append(one_word.token_length)
            self.find_contex_length()
            self.words.append(one_word)
            have_candidate = one_word.check_word2vec() # similarity ratio
            blacklist = one_word.check_blacklist()
            repeats = one_word.check_tripple_letters()
            self.test_token_list.append(int(one_word.token_length))
            self.test_blacklist.append(blacklist)
            self.test_repeated.append(repeats)
            self.test_word2vec.append(have_candidate)
            self.test_context.append(len(self.context))
            #model_prob = CORRECTOR.correct_word(one_word.word, self.sentence[:i], self.sentence[i+1:], result="prob")
            #self.lm_em.append(model_prob)
        dictionary = {'tok_len': self.test_token_list, 'blacklist': self.test_blacklist,
                      'repeats': self.test_repeated, 'word2vec': self.test_word2vec,
                      'context':self.test_context}
        sentence_matrix = self.make_test_matrix(dictionary)
        sentence_matrix = preprocessing.normalize(sentence_matrix)
        results, probabilities = predict_results(model, sentence_matrix)
        return results, probabilities


    def make_test_matrix(self, dictionary):
        data_frame = pd.DataFrame(dictionary)
        return data_frame


    def find_contex_length(self):
        self.context.append(self.token_length - 1)


def replace_bad_words(word):
        '''
        Check our cheat-dictionary
        :param word:
        :return: one word
        '''
        bad_dictionary = {'грит': 'говорит', 'тыщ': 'тысяч', 'иво': 'его', 'чо-то': 'что-то', 'че': 'что',
                          'чо': 'что', 'грю': 'говорю', 'оч': 'очень', 'че-то': 'что-то', 'ничо': 'ничего',
                          'тыш': 'тысяч', 'терь': 'теперь', 'што': 'что', 'вобще': 'вообще', 'когдато': 'когда-то',
                          'както': 'как-то', 'изза': 'из-за', 'шоб': 'чтоб', 'ващще': 'вообще', 'вопщем': 'в общем',
                          'ваще': 'вообще', 'вооще': 'вообще', 'ево': 'его', 'седня': 'сегодня', 'можт': 'может',
                          'ща': 'сейчас', 'щя': 'сейчас','ессно': 'ествественно', 'чтото': 'что-то', 'тыж': 'ты ж', 'ниче': 'ничего', 'аццки':'адски',
                          'ктото': 'кто-то', 'вобщем': 'в общем', 'вообщем': 'в общем', 'ктож': 'кто ж', 'шас': 'сейчас', 'еслиб': 'если б'}
        if word in bad_dictionary.keys():
            right_word = bad_dictionary[word]
            return right_word
        else:
            return word


def make_target_vector(errors, source):
    """
    Makes for a sentence targets (1 - misspell,0-ok )
    :param errors: array of indexes of misspels words
    :param source: array with words in a sentence
    """
    for n in range(0,len(source)):
            if n in errors:
                TARGETS.append(1)
            else:
                TARGETS.append(0)


def make_train_matrix():
    """
    Make matrix and divide on train and test
    :return: train_set, train_target, test_set, test_targets
    """
    data_frame = pd.DataFrame(TRAIN_DICTIONARY)
    X_train, y_train = data_frame, TARGETS
    return X_train, np.array(y_train)


def process_classifier(X_train, Y_train):
    """
    Classifier. Logistic regression.
    :return: predicited: array of results for test set (0 or 1), probs: probabilities for results
    """
    print("Begin classifier")
    logreg = LogisticRegression(C=1e5, class_weight = {0:0.1, 1:0.9}) #var
    logreg.fit(X_train, Y_train)
    accuraccy = logreg.score(X_train, Y_train)
    print("Accuracy", accuraccy)
    scores = cross_val_score(LogisticRegression(), X_train, Y_train, scoring='accuracy', cv=10)
    print("Cross-validation: ", scores)
    return logreg


def predict_results(logreg, X_test):
    """
    For one sentence predict results
    :param logreg: model
    :param X_test: test set, vectors
    :return:
    """
    predicted = logreg.predict(X_test)
    probs = logreg.predict_proba(X_test)
    return predicted, probs


def train_part_process():
    for num, (source, correct) in\
            enumerate(zip(source_sents, correct_sents)):
        indexes = align_sents(source, correct, return_only_different=True)
        errors = []
        for ((i, j), (k, l)) in indexes:
            errors.append(i)
        make_target_vector(errors, source)
        sentence_object = Bad_sentence(source)
        sentence_object.find_sentence_info()


def check_results_with_dic(results, source, probabilities):
    """
    Checks in Dictionary and writes the array with ok-words and *words
    :param results: 0\1 results after model for words in one sentence
    :param source: array with source words in one sentence
    :param probabilities: probabilities for words after model
    :return: after_dictionary: array with words after checking with dictionary
    """
    after_dictionary = []
    for result, word, prob in zip(results, source, probabilities):
            if result == 1:#if misspel after model
                if word in GOOD_DICTIONARY: #check the dictionary
                    after_dictionary.append(word)
                else:
                    if '-' in word: #check words with defis
                            defis_words = word.split('-')
                            in_dic = []
                            for w in defis_words: #check every word, if in Dic -> True
                                if w in GOOD_DICTIONARY:
                                    in_dic.append(True)
                                else:
                                    in_dic.append(False)
                            if False in in_dic: # if even one False => misspell
                                after_dictionary.append("*"+word)
                            else:
                                after_dictionary.append(word)
                    else: # if not with defis, not in DIC, but latin
                        if re.match('[a-z]+',word) != None:
                            after_dictionary.append(word)
                        else: #if not in DIC, not defis, not latin
                            print("Misspell:", word)
                            after_dictionary.append("*"+word)
            else: #if ok-word
                after_dictionary.append(word)
    return after_dictionary


def test_part_process(model):
    new_test_sentences = []
    for sentence in test_sents[:1000]:
            new_sentence = []
            for word in sentence:
                new_sentence.append(replace_bad_words(word))
            new_test_sentences.append(new_sentence)
    for sentence in new_test_sentences:
        sentence_object = Bad_sentence(sentence)
        results, probabilities = sentence_object.find_sentence_info_test(model) #here are results for sentence after model
        after_dictionary = check_results_with_dic(results, sentence, probabilities)
        ALL_RESULTS.append(after_dictionary)


def make_good_dictionary():
    with open('lemmas_New.txt', 'r', encoding='utf-8') as dictionary:
        dictionary_sentences = dictionary.read().split('\n')
        for words in dictionary_sentences:
            [GOOD_DICTIONARY.add(word.replace('ё','е'))for word in words.split(' ')]
    print("Dictionary for checking misspell is made")


def read_ner_data():
    with open('namedEntitiesNew.txt', 'r', encoding='utf-8') as dictionary:
        dictionary_names = dictionary.read().split('\n')
        for word in dictionary_names:
            GOOD_DICTIONARY.add(word)
    print("Dictionary for NER data is made")


def write_in_file():
    with open('result_text.text', 'w', encoding='utf-8') as result_text:
        for sentence in ALL_RESULTS:
            for word in sentence:
                result_text.write(word+' ')
            result_text.write('\n')


if __name__ == "__main__":
    with open('source_sents_new.txt', "r", encoding="utf8") as fsource,\
            open('corrected_sents_new.txt', "r", encoding="utf8") as fcorr:
        source_sents = [extract_words(line.strip())
                        for line in fsource.readlines() if line.strip() != ""]
        correct_sents = [extract_words(line.strip())
                         for line in fcorr.readlines() if line.strip() != ""]
    make_good_dictionary()
    read_ner_data()

    with open("SpellRuEval_test_sample.txt", 'r', encoding="utf-8") as test_text:
        test_sents = [extract_words(line.strip())
                        for line in test_text.readlines() if line.strip() != ""]


    #train_part = len(source_sents) * 0.7
    #train_part_process(int(train_part))
    train_part_process()
    print("Train part is ready")
    X_train, X_test = make_train_matrix()
    X_train = preprocessing.normalize(X_train)
    print("Made matrix for train data")
    model = process_classifier(X_train, X_test)
    print("Model is ready")

    test_part_process(model)
    print('Begin to write everything to testfile')
    write_in_file()
