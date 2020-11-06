# from ariadne.contrib.jieba import JiebaSegmenter
# from ariadne.contrib.nltk import NltkStemmer
# from ariadne.contrib.sbert import SbertSentenceClassifier
# from ariadne.contrib.sklearn import SklearnSentenceClassifier, SklearnMentionDetector
# from ariadne.contrib.stringmatcher import LevenshteinStringMatcher
#from ariadne.contrib.adapter import AdapterClassifier
from ariadne.contrib.bert_prediction_only import BertPredictionOnlyClassifier
from ariadne.contrib.bert import BertClassifier
from ariadne.server import Server
from ariadne.util import setup_logging
import pandas as pd
from pathlib import Path

setup_logging()

server = Server()
# server.add_classifier("spacy_ner", SpacyNerClassifier("en"))
# server.add_classifier("spacy_pos", SpacyPosClassifier("en"))
# server.add_classifier("sklearn_sentence", SklearnSentenceClassifier())
# server.add_classifier("jieba", JiebaSegmenter())
# server.add_classifier("stemmer", NltkStemmer())
# server.add_classifier("leven", LevenshteinStringMatcher())
# server.add_classifier("sbert", SbertSentenceClassifier())
# server.add_classifier("adapter", AdapterClassifier())
model_name = "bert-base-german-cased"
orig_dataset = "/ukp-storage-1/beck/Data/inception-recommenders/annotation_mace.tsv"
model_directory = Path("/ukp-storage-1/beck/Data/inception-recommenders/models")
df = pd.read_csv(orig_dataset, sep="\t")
labels = list(df["gold"].unique())
server.add_classifier("bert_prediction_only", BertPredictionOnlyClassifier(model_directory=model_directory, model_name=model_name, labels=labels))
server.add_classifier("bert", BertClassifier(model_directory=model_directory, model_name=model_name, labels=labels))

server.start(debug=True, port=40022)
