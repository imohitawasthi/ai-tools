"""
This module contains generic utilities.
"""

import random
import re

"""
Sample Cluster Generator Start
Use for generating random clusters based on user requirements.
"""


def create_random_cluster(min_range, max_range, dim, size):
    """
    Worlds most elegant cluster making function.
    Caution: Do not play with this.
    :param min_range: Minimum range of randomly generated value
    :param max_range: Maximum range of randomly generated value
    :param dim: Number of elements in Single Vector
    :param size: Entire size of the vector population
    :return: List of vectors containing all above attributes
    """
    return [[random.randint(min_range, max_range)
             for _ in range(dim)]
            for _ in range(size)]


"""
Sample Cluster Generator Ends

Tokenizer starts
Use to break text or set of texts into individual words.
"""


def tokenizer(regex, speech):
    """
    :param regex: regular expression for text extraction
    :param speech: speech which needs to be extracted
    :return: array of words which are extracted from speech
    """
    return re.findall(regex, speech)


"""
Tokenizer ends
"""
