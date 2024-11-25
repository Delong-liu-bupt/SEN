from typing import List
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from PIL import Image
from prettytable import PrettyTable
import random
import regex as re
import copy
import json
import nltk
nltk.download('averaged_perceptron_tagger') 
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')

def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:  # 替换指定数量的单词后停止
            break
    sentence = ' '.join(new_words)
    return sentence

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    sentence = ' '.join(new_words)
    return sentence

def add_word(words):
    new_word = random.choice(nltk.corpus.words.words())
    words.insert(random.randint(0, len(words) - 1), new_word)

def random_deletion(words, p):
    if len(words) == 1:
        return words
    new_words = []
    for word in words:
        if random.uniform(0, 1) > p:
            new_words.append(word)
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]
    return ' '.join(new_words)

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    sentence = ' '.join(new_words)
    return sentence

def swap_word(words):
    new_words = words.copy()
    idx1, idx2 = random.sample(range(len(new_words)), 2)
    new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

def change_sentence(sentence,p):
    result_sen = sentence
    if random.uniform(0, 1) < p:
        tokenized_sentence = nltk.word_tokenize(sentence)
        ran_num = random.random()
        if ran_num < 0.25:
            result_sen = synonym_replacement(tokenized_sentence, n=1)
        elif ran_num < 0.5:
            result_sen = random_insertion(tokenized_sentence, n=1)
        elif ran_num < 0.75:
            result_sen = random_deletion(tokenized_sentence, p=0.1)
        else:
            result_sen = random_swap(tokenized_sentence, n=1)
    return result_sen

class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("SEN.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption= self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
        }

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption

class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption ,_ ,_ = self.dataset[index]
        # pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        # pos_tags = nltk.pos_tag(caption)
        # print(caption,pos_tags)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels
        }

        return ret
    
    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405

        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()

                pos_tag = nltk.pos_tag([self.tokenizer.decoder[token][:-4]])
                if pos_tag[0][1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']:
                    # mask token with 30% probability
                    if prob < 0.3:
                        prob /= 0.3

                        # 80% randomly change token to mask token
                        if prob < 0.8:
                            tokens[i] = mask

                        # 10% randomly change token to random token
                        elif prob < 0.9:
                            tokens[i] = random.choice(token_range)

                        # -> rest 10% randomly keep current token

                        # append current token to output (we will predict these later)
                        labels.append(token)
                    else:
                        # no masking token (will be ignored by loss function later)
                        labels.append(0)
                else:
                    if prob < 0.05:
                        prob /= 0.05

                        # 80% randomly change token to mask token
                        if prob < 0.8:
                            tokens[i] = mask

                        # 10% randomly change token to random token
                        elif prob < 0.9:
                            tokens[i] = random.choice(token_range)

                        # -> rest 10% randomly keep current token

                        # append current token to output (we will predict these later)
                        labels.append(token)
                    else:
                        # no masking token (will be ignored by loss function later)
                        labels.append(0)
            else:
                labels.append(0)

        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)

class ImageTextMAEDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        # pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        # pos_tags = nltk.pos_tag(caption)
        # print(caption,pos_tags)
        if random.random() < 0.2:
            caption = self.filter_sentence(caption)

        # caption = change_sentence(caption,0.2)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())
        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
        }

        return ret
    
    def filter_sentence(self , sentence):
        tokens = word_tokenize(sentence)
        tagged_tokens = pos_tag(tokens)
        filtered_tokens = [token for token, tag in tagged_tokens if tag.startswith('PRP') or tag.startswith('NN') or tag.startswith('JJ')]
        filtered_sentence = ' '.join(filtered_tokens)
        return filtered_sentence
    
    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405

        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                pos_tag = nltk.pos_tag([self.tokenizer.decoder[token][:-4]])
                if pos_tag[0][1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']:
                    # mask token with 30% probability
                    if prob < 0.3:
                        prob /= 0.3

                        # 80% randomly change token to mask token
                        if prob < 0.8:
                            tokens[i] = mask

                        # 10% randomly change token to random token
                        elif prob < 0.9:
                            tokens[i] = random.choice(token_range)

                        # -> rest 10% randomly keep current token

                        # append current token to output (we will predict these later)
                        labels.append(token)
                    else:
                        # no masking token (will be ignored by loss function later)
                        labels.append(0)
                else:
                    if prob < 0.05:
                        prob /= 0.05

                        # 80% randomly change token to mask token
                        if prob < 0.8:
                            tokens[i] = mask

                        # 10% randomly change token to random token
                        elif prob < 0.9:
                            tokens[i] = random.choice(token_range)

                        # -> rest 10% randomly keep current token

                        # append current token to output (we will predict these later)
                        labels.append(token)
                    else:
                        # no masking token (will be ignored by loss function later)
                        labels.append(0)
            else:
                labels.append(0)

        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)