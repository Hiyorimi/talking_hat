from Phonetic import Letter


def count_double_letter(word, letter_type='vowel', debug=False):
    """
    Counts occurrences of certain letter type in the given word.
    :param word: string
    :param letter_type: 'vowel' or 'consonant'
    :param debug: debug flag
    :return:
    """

    if len(word) == 0:
        return 0
    counter = 0
    for i in range(0, len(word)-1):
        try:
            letter = word[i]
            if Letter(letter).classify()[letter_type]:
                next_letter = word[i+1]
                if Letter(next_letter).classify()[letter_type]:
                    counter += 1
        except Exception:
            if debug:
                print (">> ", word, letter)
            pass

    return counter


def starts_with_letter(word, letter_type='vowel'):
    """
    Returns class of the letter, word start with.
    :param word: input word
    :param letter_type: 'vowel' or 'consonant'
    :return: Boolean
    """

    if len(word) == 0:
        return False
    return Letter(word[0]).classify()[letter_type]

def ends_with_letter(word, letter_type='vowel', debug=False):
    """
    Returns class of the letter, word ends with.
    :param word: input word
    :param letter_type: 'vowel' or 'consonant'
    :param debug: debug flag
    :return: Boolean
    """
    try:
        if len(word) == 0:
            return False
        return Letter(word[len(word)-1]).classify()[letter_type]
    except Exception:
        if debug:
            print(">>", word)
        return False


def count_letter_type(word, debug=False):
    """
    Counts number of occurrences of different letter types in the given word.
    :param word: input word
    :param debug: debug flag
    :return: :obj:`dict` of :obj:`str` => :int:count
    """
    count = {
         'consonant': 0,
         'deaf': 0,
         'hard': 0,
         'mark': 0,
         'paired': 0,
         'shock': 0,
         'soft': 0,
         'sonorus': 0,
         'vowel': 0
    }
    try:
        for letter in word:
            classes = Letter(letter).classify()
            for key in count.keys():
                if classes[key]:
                    count[key] += 1
    except Exception:
        if debug:
            print (">>", letter)
        pass
    return count


