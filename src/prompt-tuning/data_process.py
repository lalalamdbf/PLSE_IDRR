import tokenization_word as tokenization
import os
from prompt.data_utils import InputExample
from prompt.pipeline_base import PromptDataLoader
import numpy as np
import math

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        file_in = open(input_file, "rb")
        lines = []
        for line in file_in:
            lines.append(line.decode("utf-8").split("\t"))
        return lines


class PromptProcessor(DataProcessor):
    def __init__(self):
        self.labels = set()

    def get_examples(self, data_dir, data_type, num_rels):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_dir), data_type, num_rels)

    def get_labels(self):
        """See base class."""
        tmp = list(self.labels)
        tmp.sort()
        return tmp

    def _create_examples(self, lines, set_type, num_rels):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            
            label = tokenization.convert_to_unicode(line[2])
            text_b = tokenization.convert_to_unicode(line[1])
            
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b,
                             label=create_multi_label(label, num_rels)))
        return examples


class DataGenerate(object):

    def get_train_loader(self):
        raise NotImplementedError()

    def get_dev_loader(self):
        raise NotImplementedError()

    def get_test_loader(self):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()


class PromptDataGenerate(DataGenerate):
    def __init__(self, args, prompt_config):

        self.args = args
        self.processors = PromptProcessor()
        self.train_path = os.path.join(args.data_dir, "train.tsv")
        self.dev_path = os.path.join(args.data_dir, "dev.tsv")
        self.test_path = os.path.join(args.data_dir, "test.tsv")
        self.blind_path = os.path.join(args.data_dir, "blind.tsv")
        self.num_train_steps = None
        self.test_data_loader = None
        self.train_data_loader = None
        self.dev_data_loader = None
        self.blind_data_loader = None
        self.tokenizer =  prompt_config.get_tokenizer()
        self.promptTemplate = prompt_config.get_template()
        self.wrappeer_class = prompt_config.get_wrapperclass()

    def get_train_loader(self):
        train_examples = self.processors.get_examples(self.train_path, "train", self.args.num_rels)

        self.num_train_steps = len(train_examples)
        self.train_data_loader = PromptDataLoader(
            dataset=train_examples,
            tokenizer=self.tokenizer,
            template=self.promptTemplate,
            max_seq_length=self.args.max_seq_length,
            tokenizer_wrapper_class=self.wrappeer_class,
            create_token_type_ids=True,
            batch_size=self.args.train_batch_size,
            shuffle=True
        )
        return self.train_data_loader

    def get_dev_loader(self):
        if self.dev_data_loader is None:
            dev_examples = self.processors.get_examples(self.dev_path, "dev", self.args.num_rels)
            self.dev_data_loader = PromptDataLoader(
                dataset=dev_examples,
                tokenizer=self.tokenizer,
                template=self.promptTemplate,
                max_seq_length=self.args.max_seq_length,
                tokenizer_wrapper_class=self.wrappeer_class,
                create_token_type_ids=True,
                batch_size=self.args.dev_batch_size,
                shuffle=False
            )
        return self.dev_data_loader
    
    def get_test_loader(self):
        if self.test_data_loader is None:
            test_examples = self.processors.get_examples(self.test_path, "test", self.args.num_rels)
            self.test_data_loader = PromptDataLoader(
                dataset=test_examples,
                tokenizer=self.tokenizer,
                template=self.promptTemplate,
                max_seq_length=self.args.max_seq_length,
                tokenizer_wrapper_class=self.wrappeer_class,
                create_token_type_ids=True,
                batch_size=self.args.test_batch_size,
                shuffle=False
            )
        return self.test_data_loader
    
    def get_blind_loader(self):
        if self.blind_data_loader is None:
            blind_examples = self.processors.get_examples(self.blind_path, "blind", self.args.num_rels)
            self.blind_data_loader = PromptDataLoader(
                dataset=blind_examples,
                tokenizer=self.tokenizer,
                template=self.promptTemplate,
                max_seq_length=self.args.max_seq_length,
                tokenizer_wrapper_class=self.wrappeer_class,
                create_token_type_ids=True,
                batch_size=self.args.test_batch_size,
                shuffle=False
            )
        return self.blind_data_loader



def create_multi_label(label_str, num_rels):
    label_list = label_str.split('#')
    if num_rels == 4:
        label_multi = [0, 0, 0, 0]
    elif num_rels == 11:
        label_multi = np.zeros(11,dtype=int)
    else:
        label_multi = np.zeros(14,dtype=int)
    for idx in label_list:
        label_multi[int(idx)] = 1
    return label_multi


