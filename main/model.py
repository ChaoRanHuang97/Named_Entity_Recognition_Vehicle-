
import spacy
from tqdm import tqdm
from spacy.training.example import Example
from spacy.scorer import Scorer
from spacy.util import minibatch, compounding
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from thinc.api import SGD
from thinc.api import decaying

optimizer_SGD = SGD(
    learn_rate=decaying(0.005, 1e-4),
    L2=1e-6,
    grad_clip=1.0
)

class ner_nlp():
    def __init__(self, iter=None,
                 TRAINING_DATA=None,
                 minibatch_size=compounding(2., 16., 1.05),
                 SGD_type=optimizer_SGD,
                 dropout = 0.3,
                 model_path=None):

        if model_path != None:
            self.nlp = spacy.load(model_path)
        else:
            self.iter = iter
            self.TRAINING_DATA = TRAINING_DATA
            self.minibatch_size = minibatch_size
            self.SGD_type = SGD_type
            self.dropout = dropout
            self.nlp = spacy.blank("en")
            # ner = self.nlp.create_pipe("ner")
            self.nlp.add_pipe('ner')
        inf = list(self.nlp.Defaults.infixes)
        inf = [x for x in inf if
               '-|–|—|--|---|——|~' not in x]  # remove the hyphen-between-letters pattern from infix patterns
        infix_re = compile_infix_regex(tuple(inf))

        def custom_tokenizer(nlp):
            return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                             suffix_search=nlp.tokenizer.suffix_search,
                             infix_finditer=infix_re.finditer,
                             token_match=nlp.tokenizer.token_match,
                             rules=nlp.Defaults.tokenizer_exceptions)

        self.nlp.tokenizer = custom_tokenizer(self.nlp)

    def train(self):
        self.nlp.begin_training()
        for itn in tqdm(range(self.iter)):
            # Shuffle the training data
            losses = {}
            # Batch the examples and iterate over them
            for batch in minibatch(self.TRAINING_DATA, size=self.minibatch_size):
                for text, annotations in batch:
                    doc = self.nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    # Update the model
                    self.nlp.update([example], losses=losses, sgd=self.SGD_type, drop=self.dropout)
            print(losses)

    def detect(self, posting):
        doc = self.nlp(posting)
        prediction = dict()
        for ent in doc.ents:
            if ent.label_ not in prediction:
                prediction[ent.label_] = ent.text
        return prediction

    def get_score(self, TESTING_DATA):
        examples = []
        scorer = Scorer(self.nlp)
        for text, annotations in TESTING_DATA:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            example.predicted = self.nlp(str(example.predicted))
            examples.append(example)
        return scorer.score(examples)

    def save_model(self, version):
        self.nlp.to_disk('models/mca-ner-draft_{}.model'.format(version))