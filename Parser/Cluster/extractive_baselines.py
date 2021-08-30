from multiprocessing import Pool
from collections import defaultdict


def check_list_of_strings(l):
    if not isinstance(l, list):
        return False
    try:
        isinstance(l[-1], str)
    except:
        return False
    return True


def lcs(a, b):
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            table[i][j] = (table[i - 1][j - 1] + 1 if ca == cb else
                           max(table[i][j - 1], table[i - 1][j]))
    return table[-1][-1]


def calc_rouge_traits(l, n, min_ngram_length=0):
    def generate_n_gram(text_l, n):
        return ' '.join([text_l[j].lower() for j in range(n)])

    count_set = {}
    # ngrams=map(generate_n_gram, [l[k:k+n] for k in range(len(l)-n)], [n]*(len(l)-n))
    for i in range(len(l) - n):
        item = ' '.join([l[i + j].lower() for j in range(n)])
        if len(item) >= min_ngram_length:
            count_set.__setitem__(item, 1 + count_set.get(item, 0))
    # ngrams=[' '.join([l[i+j].lower() for j in range(n)]) for i in range(len(l)) if i+n <= len(l)]
    # for item in ngrams:
    #    count_set.__setitem__(item, 1 + count_set.get(item, 0))
    return count_set


def rouge_n(a, b, n=1, min_ngram_length=0):
    assert check_list_of_strings(a), 'Object "a" should be list of strings'
    assert check_list_of_strings(b), 'Object "b" should be list of strings'
    assert len(b) > 0, 'List "b" should have at least one element'
    assert len(a) > 0, 'List "a" should have at least one element'
    ref = calc_rouge_traits(b, n, min_ngram_length)
    sys = calc_rouge_traits(a, n, min_ngram_length)
    inter = [min(ref.get(k, 0), sys.get(k, 0)) for k in ref.keys()]
    rec = sum(inter) / sum(ref.values()) if sum(ref.values()) > 0 else 0
    pr = sum(inter) / sum(sys.values()) if sum(sys.values()) > 0 else 0
    if rec + pr == 0:
        return 0, 0, 0
    fs = 2 * rec * pr / (rec + pr)
    return pr, rec, fs


def rouge_l(a, b):
    assert check_list_of_strings(a), 'Object "a" should be list of strings'
    assert check_list_of_strings(b), 'Object "b" should be list of strings'
    assert len(b) > 0, 'List "b" should have at least one element'
    assert len(a) > 0, 'List "a" should have at least one element'
    ref = calc_rouge_traits(b, 1)
    sys = calc_rouge_traits(a, 1)
    inter = lcs([w.lower() for w in a], [w.lower() for w in b])
    rec = inter / sum(ref.values())
    pr = inter / sum(sys.values())
    if rec + pr == 0:
        return 0, 0, 0
    fs = 2 * pr * rec / (pr + rec)
    return pr, rec, fs


def find_oracle(tgt, src, tmp=[], choices=[], depth=3, n=2, mplevel=-1, score='f1'):
    """Performs exaustive search to find best possible extractive summary based on abstractive summary.

    Parameters:
        tgt: list[string]
            Tokenized abstractive summary
        src: list[list[string]]
            Sentence-wise tokenized text
        depth: int, default=3
            Maximum number of extracted sentences
        n: int, default=2
            Type of rouge_n metric
        mplevel: int, optional
            Tree level where multiprocessing should be applied (int)
        score: {'precision', 'recall', or 'f1'}, default='f1'
            Rouge_n score type

    """
    if mplevel < 0:
        mplevel = -1

    s_types = {"precision": 0, "recall": 1, "f1": 2}

    if len(src) == 0 or len(tmp) == depth:
        if len(tmp) == 0:
            return ([''], choices, 0)
        text = [j for i in tmp for j in i]
        return (text, choices, rouge_n(text, tgt, n=n)[s_types[score]])
    if mplevel == 0:
        pool = Pool()
        res1 = pool.apply_async(find_oracle, (tgt, src[1:], tmp + [src[0]], choices + [1], depth, n, -1))
        res2 = pool.apply_async(find_oracle, (tgt, src[1:], tmp, choices + [0], depth, n, -1))
        a = res1.get()
        b = res2.get()
    else:
        a = find_oracle(tgt, src[1:], tmp + [src[0]], choices + [1], depth, n, mplevel - 1)
        b = find_oracle(tgt, src[1:], tmp, choices + [0], depth, n, mplevel - 1)

    if a[2] > b[2]:
        return a
    elif a[2] == b[2] and n > 1 and rouge_n(a[0], tgt, n=n - 1)[s_types[score]] > rouge_n(b[0], tgt, n=n - 1)[
        s_types[score]]:
        return a
    else:
        return b


def find_oracle_combined(tgt, src, tmp=[], choices=[], depth=3, ns=[1, 2], agg='sum', mplevel=-1, score='f1'):
    """Performs exaustive search to find best possible extractive summary based on abstractive summary.

    Parameters:
        tgt: list[string]
            Tokenized abstractive summary
        src: list[list[string]]
            Sentence-wise tokenized text
        depth: int, default=3
            Maximum number of extracted sentences
        ns: list[int], default=[1,2]
            Type of rouge_n metric to combine
        mplevel: int, optional
            Tree level where multiprocessing should be applied (int)
        score: {'precision', 'recall', or 'f1'}, default='f1'
            Rouge_n score type

    """
    if mplevel < 0:
        mplevel = -1

    s_types = {"precision": 0, "recall": 1, "f1": 2}
    agg_types = {'sum': sum}

    if len(src) == 0 or len(tmp) == depth:
        if len(tmp) == 0:
            return ([''], choices, [0] * len(ns))
        text = [j for i in tmp for j in i]
        sc_n = [rouge_n(text, tgt, n=n)[s_types[score]] for n in ns]
        return (text, choices, sc_n)
    if mplevel == 0:
        pool = Pool()
        res1 = pool.apply_async(find_oracle_combined, (tgt, src[1:], tmp + [src[0]], choices + [1], depth, ns, agg, -1))
        res2 = pool.apply_async(find_oracle_combined, (tgt, src[1:], tmp, choices + [0], depth, ns, agg, -1))
        a = res1.get()
        b = res2.get()
    else:
        a = find_oracle_combined(tgt, src[1:], tmp + [src[0]], choices + [1], depth, ns, agg, mplevel - 1)
        b = find_oracle_combined(tgt, src[1:], tmp, choices + [0], depth, ns, agg, mplevel - 1)

    if agg_types[agg](a[2]) > agg_types[agg](b[2]):
        return a
    elif agg_types[agg](a[2]) > agg_types[agg](b[2]) and rouge_n(a[0], tgt, n=ns[0])[s_types[score]] > \
            rouge_n(b[0], tgt, n=ns[0])[s_types[score]]:
        return a
    else:
        return b


if __name__ == '__main__':
    a = 'under the bed the cat was found'.split()
    b = 'the cat was under the bed'.split()
    print(a, b)
    print(rouge_n(a, b, n=1))
    print(rouge_n(a, b, n=2))
    print(rouge_l(a, b))
