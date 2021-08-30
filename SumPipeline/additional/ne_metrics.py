from . import extractive_baselines as eb
import spacy

path_to_rumodel='../ru2_combined_400ks_96'

default_nlp = spacy.load('ru2_combined_400ks_96')
default_nlp.add_pipe(default_nlp.create_pipe('sentencizer'), first=True)


class NE_compare:
    def __init__(self, nlp=default_nlp, token_vise=False):
        self.nlp = nlp
        self.token_vise = token_vise

    def process_src(self, src):
        self.src_feats = self.get_NE(src)

    def get_NE(self, text):
        parsed = self.nlp(text)
        return list(set(e.text for e in parsed.ents))

    def NE_overlap(self, a: str, report_token_vise=False):
        a_ne = self.get_NE(a)
        b_ne = self.src_feats

        if report_token_vise:
            a_ne = sum([i.split() for i in a_ne], [])
            b_ne = sum([i.split() for i in b_ne], [])
        if not a_ne: a_ne = [" "]
        if not b_ne: b_ne = [" "]
        return eb.rouge_n(a_ne, b_ne, return_dict=True)

    def compare(self, cand):
        score = self.NE_overlap(cand, report_token_vise=self.token_vise)
        return score['Precision']


class NEF_compare:
    def __init__(self, nlp=default_nlp):
        self.nlp = nlp

    def process_src(self, src):
        self.src_feats = self.get_relations(src)


    def get_relations(self, text):
        rels = []
        res = self.nlp(text)
        rels += list(set(e.lemma_ for e in res.ents))

        for sent in res.sents:
            for w in sent:
                for c in w.children:
                    if ("mod" in c.dep_ or "pos" in c.dep_) and (w.ent_type or c.ent_type):
                        rels.append((w.lemma_, 'is', c.lemma_))
                    if "subj" in c.dep_ or "comp" in c.dep_ or "obj" in c.dep_:
                        rels.append((w.lemma_, 'is', c.lemma_))

        return list(set([" ".join(r) for r in rels]))

    def compare(self, cand):
        a_ne = self.get_relations(cand)
        b_ne = self.src_feats
        if not a_ne: a_ne = [" "]
        if not b_ne: b_ne = [" "]
        score = eb.rouge_n(a_ne, b_ne, return_dict=True)
        return score['Precision']

