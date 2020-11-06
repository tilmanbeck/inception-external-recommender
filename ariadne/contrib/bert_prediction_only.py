import logging

from cassis import Cas
from diskcache import Cache
import pandas as pd
from transformers import AutoTokenizer, InputFeatures, AutoModelForSequenceClassification, \
    TrainingArguments, InputExample, DataProcessor
from typing import List, Optional
from .OptimTrainer import OptimTrainer
import torch
from enum import Enum
import numpy as np
from pathlib import Path
from ariadne import cache_directory
from ariadne.classifier import Classifier
from ariadne.protocol import TrainingDocument
from dataclasses import dataclass, field
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    data_file: str = field(
        default=None,
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    task_name: str = field(default="Twitter")
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: AutoTokenizer,
    mode: Split,
    max_length: Optional[int] = None,
    label_list: List = None,
    output_mode="classification",
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True
):

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)

        if output_mode == "classification":
            if mode in [Split.dev, Split.train]:
                label = label_map[example.label]
            else:
                label = None
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            if label is not None:
                logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label))

    return features

class Processor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def __init__(self, labels, data):
        self.labels = labels
        self.data = data

    def get_examples(self, mode="train"):
        """See base class."""
        return self._create_examples(self.data, mode)

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, data, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == Split.test
        examples = []
        for i, (text, label) in enumerate(zip(data["texts"], data["labels"])):
            guid = "%s-%s" % (set_type, i)
            text_a = text
            label = None if test_mode else label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class TwitterDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: DataTrainingArguments
    output_mode: str
    features: List[InputFeatures]
    mode: Split

    def __init__(
        self,
        args: DataTrainingArguments,
        tokenizer: AutoTokenizer,
        processor: DataProcessor,
        mode: Split,
        limit_length: Optional[int] = None,
    ):
        self.args = args
        self.processor = processor
        self.output_mode = "classification"
        self.mode = mode

        label_list = self.processor.get_labels()
        self.label_list = label_list

        logger.info(f"Creating features from dataset")

        examples = self.processor.get_examples()
        if limit_length is not None:
            examples = examples[:limit_length]
        self.features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            label_list=label_list,
            output_mode=self.output_mode,
            mode=self.mode
        )
#        start = time.time()
#        torch.save(self.features, cached_features_file)
        # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
#        logger.info(
#            "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
#        )
    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]



training_args = TrainingArguments(
    logging_steps=100,
    per_device_train_batch_size=8,
#            per_device_eval_batch_size=64,
    save_steps=1000,
#            evaluate_during_training=True,
    evaluate_during_training=False,
    output_dir="model_output/",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=False,
    do_predict=False,
    learning_rate=0.0008,
    num_train_epochs=5
)
data_args = DataTrainingArguments(
    overwrite_cache=True
)


class CachedTokenizer:
    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_tokenizer(self):
        return self.tokenizer

class BertPredictionOnlyClassifier(Classifier):

    tokenizer = None
    labels = None

    def __init__(self, model_directory: Path, model_name: str, labels: List[str]):
        super().__init__(model_directory=model_directory)
        self.tokenizer = CachedTokenizer(model_name).get_tokenizer()
        self.labels = labels


    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        logger.debug("No training performed")

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        model = self._load_model("twitter_opinion_bert_finetuned")

        if model is None:
            logger.debug("No trained model ready yet!")
            return

        sentences_cas = self.iter_sentences(cas)
        sentences = [s.get_covered_text() for s in self.iter_sentences(cas)]

        processor = Processor(labels=self.labels, data={"texts": sentences, "labels": [None for i in range(len(sentences))]})
        test_dataset = (
            TwitterDataset(data_args, tokenizer=self.tokenizer, processor=processor, mode=Split.test)
        )
        trainer = OptimTrainer(
            model=model,
            args=training_args
        )
        predictions = trainer.predict(test_dataset=test_dataset).predictions
        predictions = np.argmax(predictions, axis=1)
        label_map = {i: label for i, label in enumerate(self.labels)}
        predictions = [label_map[i] for i in predictions]

        for sentence, label in zip(sentences_cas, predictions):
            prediction = self.create_prediction(cas, layer, feature, sentence.begin, sentence.end, label)
            cas.add_annotation(prediction)

