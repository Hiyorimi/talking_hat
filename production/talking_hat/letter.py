# FROM: https://github.com/NyashniyVladya/RusPhonetic/blob/master/RusPhonetic/phonetic_module.py
def Letter(s):
    if isEnglish(s):
        return EnglishLetter(s)
    else:
        return RussianLetter(s)
    
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


class RussianLetter(object):

    """
    Класс буквы.
    """

    vowels = "аеёиоуыэюя"  # Гласные буквы
    consonants = "бвгджзйклмнпрстфхцчшщ"  # Согласные буквы
    marks = "ъь"  # Знаки

    forever_hard = "жшц"  # Всегда твёрдые.
    forever_soft = "йчщ"  # Всегда мягкие.

    vovels_set_hard = "аоуыэ"  # Делают предыдущую согласную твёрдой.
    vovels_set_soft = "еёиюя"  # Делают предыдущую согласную мягкой.

    ioted_vowels = {  # Йотированные гласные.
        'е': 'э',
        'ё': 'о',
        'ю': 'у',
        'я': 'а'
    }

    forever_sonorus = "йлмнр"  # Всегда звонкие.
    forever_deaf = "xцчщ"  # Всегда глухие.

    sonorus_deaf_pairs = (  # Пары звонкий-глухой.
        ('б', 'п'),
        ('в', 'ф'),
        ('г', 'к'),
        ('д', 'т'),
        ('ж', 'ш'),
        ('з', 'с')
    )

    def __init__(self, letter, prev_letter=None, shock=False):
        """
        :letter:
            Сама буква.
        :prev_letter:
            Предыдущая буква в слове, если есть.
        :shock:
            Если гласная, то ударная ли.
        """

        if prev_letter is not None:
            if not isinstance(prev_letter, self.__class__):
                raise Exception(
                    (
                        "Предыдущая буква должна быть объектом класса {0!r}, "
                        "или None (передан тип {1!r})."
                    ).format(self.__class__, prev_letter.__class__)
                )

        self.__letter = letter.lower().strip()
        self.__prev_letter = prev_letter

        if len(self.__letter) != 1:
            raise Exception("Передано неверное количество символов.")

        if not (self.is_vowel() or self.is_consonant() or self.is_mark()):
            raise Exception("Передана не буква русского языка.")

        self.__shock = (self.is_vowel() if shock else False)

        self.__forced_hard = None
        self.__forsed_sonorus = None
        self._forced_not_show = False
        self._is_double = False

        self.set_prev_sonorus()
        self.set_prev_hard()
        self.set_double_sound()

    def set_double_sound(self):
        prev = self.get_prev_letter()
        if not prev:
            return
        prev._forced_not_show = False
        prev._is_double = False
        self._is_double = False
        prev.set_double_sound()
        if self.is_consonant() and prev.is_consonant():
            if self._get_sound() == prev._get_sound():
                prev._forced_not_show = True
                prev._is_double = True
                self._is_double = True

    def set_prev_sonorus(self):
        """
        Выставляет параметры звонкости/глухости, для предыдущих согласных.
        """
        prev = self.get_prev_letter()
        if not prev:
            return
        if not (self.is_consonant() and prev.is_consonant()):
            return
        if self.is_sonorus() and self.is_paired_consonant():
            if self._get_sound(False) != 'в':
                prev.set_sonorus(True)
            return
        if self.is_deaf():
            prev.set_sonorus(False)
            return

    def set_prev_hard(self):
        """
        Выставляет параметры твёрдости/мягкости, для предыдущих согласных.
        """
        prev = self.get_prev_letter()
        if not prev:
            return
        if not prev.is_consonant():
            return
        if self.is_softener(prev):
            prev.set_hard(False)
        elif self.letter in self.vovels_set_hard:
            prev.set_hard(True)

    def is_after_acc(self):
        """
        Буква распологается после ударения.
        """
        prev = self._prev_letter()
        while True:
            if not prev:
                return False
            if prev.is_shock():
                return True
            prev = prev._prev_letter()

    def get_sound(self):
        if self.is_mark() or self._forced_not_show:
            return ""
        _snd = self._get_sound()
        if self._is_double and self.is_after_acc():
            _snd += ":"
        return _snd

    def _get_sound(self, return_soft_mark=True):

        if self.is_mark():
            return ""

        prev = self._prev_letter()
        _letter_now = self.letter

        if self.is_vowel():
            if _letter_now in self.ioted_vowels.keys():

                _let = self.ioted_vowels[_letter_now]
                if (not prev) or prev.is_vowel() or prev.is_mark():
                    _letter_now = "й'{0}".format(_let)
                elif not self.is_shock():
                    _letter_now = 'и'
                else:
                    _letter_now = _let

            if _letter_now == 'о':
                if not self.is_shock():
                    _letter_now = 'а'

            if (_letter_now == 'и') and prev:
                if prev.letter == 'ь':
                    _letter_now = "й'и"
                elif prev.letter in prev.forever_hard:
                    _letter_now = 'ы'
            return _letter_now

        _let = self.get_variant(self.is_deaf())
        if return_soft_mark and self.is_soft():
            _let += "'"
        return _let

    def initialize_as_end(self):
        if self.is_consonant():
            self.set_sonorus(False)

    def set_hard(self, new_value=None):
        if self.letter in (self.forever_hard + self.forever_soft):
            return
        self.__forced_hard = new_value
        self.set_prev_hard()

    def set_sonorus(self, new_value=None):
        self.__forsed_sonorus = new_value
        self.set_prev_sonorus()

    @property
    def letter(self):
        return self.__letter

    def get_prev_letter(self):
        """
        Возвращает предыдущий объект буквы, если она не является знаком.
        Если знак, то рекурсивно спускается, до ближайшей.
        """
        prev = self._prev_letter()
        while True:
            if not prev:
                return prev
            if prev.letter in prev.marks:
                prev = prev._prev_letter()
                continue
            return prev

    def _prev_letter(self):
        """
        Возвращает предыдущую букву, без особых указаний.
        """
        return self.__prev_letter

    def get_variant(self, return_deaf):
        """
        Возвращает вариант буквы.
        :return_deaf:
            True - вернуть глухой вариант. Если False - звонкий.
        """
        return_deaf = bool(return_deaf)
        for variants in self.sonorus_deaf_pairs:
            if self.__letter in variants:
                return variants[return_deaf]
        return self.__letter

    def is_paired_consonant(self):
        """
        Парная ли согласная.
        """
        if not self.is_consonant():
            return False
        for variants in self.sonorus_deaf_pairs:
            if self.letter in variants:
                return True
        return False

    def is_sonorus(self):
        """
        Звонкая ли согласная.
        """
        if not self.is_consonant():
            return False
        if self.letter in self.forever_sonorus:
            return True
        if self.letter in self.forever_deaf:
            return False
        if self.__forsed_sonorus:
            return True
        if self.__forsed_sonorus is False:
            return False
        for son, _ in self.sonorus_deaf_pairs:
            if self.letter == son:
                return True
        return False

    def is_deaf(self):
        """
        Глухая ли согласная.
        """
        if not self.is_consonant():
            return False
        if self.letter in self.forever_deaf:
            return True
        if self.letter in self.forever_sonorus:
            return False
        if self.__forsed_sonorus:
            return False
        if self.__forsed_sonorus is False:
            return True
        for _, df in self.sonorus_deaf_pairs:
            if self.letter == df:
                return True
        return False

    def is_hard(self):
        if not self.is_consonant():
            return False
        if self.letter in self.forever_hard:
            return True
        if self.letter in self.forever_soft:
            return False
        if self.__forced_hard:
            return True
        return False

    def is_soft(self):
        if not self.is_consonant():
            return False
        if self.letter in self.forever_soft:
            return True
        if self.letter in self.forever_hard:
            return False
        if self.__forced_hard is False:
            return True
        return False

    def end(self, string):
        """
        Проверяет, заканчивается ли последовательность букв переданной строкой.
        Скан производится, без учёта текущей.
        """
        prev = self._prev_letter()
        for s in reversed(string):
            if prev.letter != s:
                return False
            if not prev:
                return False
            prev = prev._prev_letter()
        return True

    def is_softener(self, let):
        """
        Является ли символ смягчающим.
        :let: Объект буквы, которую пытаемся смягчить.
        """
        if let.letter in let.forever_hard:
            return False
        if not let.is_consonant():
            return False
        if self.letter in self.vovels_set_soft:
            return True
        if self.letter == 'ь':
            return True
        if self.is_soft() and (let.letter in "дзнст"):
            return True
        if self.letter == 'ъ':
            if self.end("раз") or self.end("из") or self.end("с"):
                return True
        return False

    def is_vowel(self):
        return (self.letter in self.vowels)

    def is_consonant(self):
        return (self.letter in self.consonants)

    def is_mark(self):
        return (self.letter in self.marks)

    def is_shock(self):
        return self.__shock
    
    def classify(self):
        return {
            'paired': self.is_paired_consonant(),
            'sonorus': self.is_sonorus(),
            'deaf': self.is_deaf(),
            'hard': self.is_hard(),
            'soft': self.is_soft(),
            'vowel': self.is_vowel(),
            'consonant': self.is_consonant(),
            'mark': self.is_mark(),
            'shock': self.is_shock()
        }


# FROM: https://github.com/NyashniyVladya/RusPhonetic/blob/master/RusPhonetic/phonetic_module.py
class EnglishLetter(object):

    """
    Класс ENG буквы.
    """

    vowels = "aeiou"  # Гласные буквы A, E, I, O, U
    # B, C, D, F, G, H, J, K, L, M, N, P, Q, R, S, T, V, W, X, Y (sometimes), and Z
    consonants = "bcdfghjklmnpqrstvwxyz"  # Согласные буквы 
    marks = ""  # Знаки

    forever_hard = "bdgzv"  # Всегда твёрдые.
    forever_soft = "cfhklmpt"  # Всегда мягкие.

    vovels_set_hard = "i"  # Делают предыдущую согласную твёрдой.
    vovels_set_soft = "qu"  # Делают предыдущую согласную мягкой.

    ioted_vowels = {  # Йотированные гласные.
        'e': 'e',
    }

    forever_sonorus = "zdt"  # Всегда звонкие.
    forever_deaf = "hfs"  # Всегда глухие.

    sonorus_deaf_pairs = (  # Пары звонкий-глухой.
        ('b', 'p'),
    )

    def __init__(self, letter, prev_letter=None, shock=False):
        """
        :letter:
            Сама буква.
        :prev_letter:
            Предыдущая буква в слове, если есть.
        :shock:
            Если гласная, то ударная ли.
        """

        if prev_letter is not None:
            if not isinstance(prev_letter, self.__class__):
                raise Exception(
                    (
                        "Предыдущая буква должна быть объектом класса {0!r}, "
                        "или None (передан тип {1!r})."
                    ).format(self.__class__, prev_letter.__class__)
                )

        self.__letter = letter.lower().strip()
        self.__prev_letter = prev_letter

        if len(self.__letter) != 1:
            raise Exception("Передано неверное количество символов.")

        if not (self.is_vowel() or self.is_consonant() or self.is_mark()):
            raise Exception("Передана не буква русского языка.")

        self.__shock = (self.is_vowel() if shock else False)

        self.__forced_hard = None
        self.__forsed_sonorus = None
        self._forced_not_show = False
        self._is_double = False

        self.set_prev_sonorus()
        self.set_prev_hard()
        self.set_double_sound()

    def set_double_sound(self):
        prev = self.get_prev_letter()
        if not prev:
            return
        prev._forced_not_show = False
        prev._is_double = False
        self._is_double = False
        prev.set_double_sound()
        if self.is_consonant() and prev.is_consonant():
            if self._get_sound() == prev._get_sound():
                prev._forced_not_show = True
                prev._is_double = True
                self._is_double = True

    def set_prev_sonorus(self):
        """
        Выставляет параметры звонкости/глухости, для предыдущих согласных.
        """
        prev = self.get_prev_letter()
        if not prev:
            return
        if not (self.is_consonant() and prev.is_consonant()):
            return
        if self.is_sonorus() and self.is_paired_consonant():
            if self._get_sound(False) != 'в':
                prev.set_sonorus(True)
            return
        if self.is_deaf():
            prev.set_sonorus(False)
            return

    def set_prev_hard(self):
        """
        Выставляет параметры твёрдости/мягкости, для предыдущих согласных.
        """
        prev = self.get_prev_letter()
        if not prev:
            return
        if not prev.is_consonant():
            return
        if self.is_softener(prev):
            prev.set_hard(False)
        elif self.letter in self.vovels_set_hard:
            prev.set_hard(True)

    def is_after_acc(self):
        """
        Буква распологается после ударения.
        """
        prev = self._prev_letter()
        while True:
            if not prev:
                return False
            if prev.is_shock():
                return True
            prev = prev._prev_letter()

    def get_sound(self):
        if self.is_mark() or self._forced_not_show:
            return ""
        _snd = self._get_sound()
        if self._is_double and self.is_after_acc():
            _snd += ":"
        return _snd

    def _get_sound(self, return_soft_mark=True):

        if self.is_mark():
            return ""

        prev = self._prev_letter()
        _letter_now = self.letter

        if self.is_vowel():
            if _letter_now in self.ioted_vowels.keys():

                _let = self.ioted_vowels[_letter_now]
                if (not prev) or prev.is_vowel() or prev.is_mark():
                    _letter_now = "й'{0}".format(_let)
                elif not self.is_shock():
                    _letter_now = 'и'
                else:
                    _letter_now = _let

            if _letter_now == 'о':
                if not self.is_shock():
                    _letter_now = 'а'

            if (_letter_now == 'и') and prev:
                if prev.letter == 'ь':
                    _letter_now = "й'и"
                elif prev.letter in prev.forever_hard:
                    _letter_now = 'ы'
            return _letter_now

        _let = self.get_variant(self.is_deaf())
        if return_soft_mark and self.is_soft():
            _let += "'"
        return _let

    def initialize_as_end(self):
        if self.is_consonant():
            self.set_sonorus(False)

    def set_hard(self, new_value=None):
        if self.letter in (self.forever_hard + self.forever_soft):
            return
        self.__forced_hard = new_value
        self.set_prev_hard()

    def set_sonorus(self, new_value=None):
        self.__forsed_sonorus = new_value
        self.set_prev_sonorus()

    @property
    def letter(self):
        return self.__letter

    def get_prev_letter(self):
        """
        Возвращает предыдущий объект буквы, если она не является знаком.
        Если знак, то рекурсивно спускается, до ближайшей.
        """
        prev = self._prev_letter()
        while True:
            if not prev:
                return prev
            if prev.letter in prev.marks:
                prev = prev._prev_letter()
                continue
            return prev

    def _prev_letter(self):
        """
        Возвращает предыдущую букву, без особых указаний.
        """
        return self.__prev_letter

    def get_variant(self, return_deaf):
        """
        Возвращает вариант буквы.
        :return_deaf:
            True - вернуть глухой вариант. Если False - звонкий.
        """
        return_deaf = bool(return_deaf)
        for variants in self.sonorus_deaf_pairs:
            if self.__letter in variants:
                return variants[return_deaf]
        return self.__letter

    def is_paired_consonant(self):
        """
        Парная ли согласная.
        """
        if not self.is_consonant():
            return False
        for variants in self.sonorus_deaf_pairs:
            if self.letter in variants:
                return True
        return False

    def is_sonorus(self):
        """
        Звонкая ли согласная.
        """
        if not self.is_consonant():
            return False
        if self.letter in self.forever_sonorus:
            return True
        if self.letter in self.forever_deaf:
            return False
        if self.__forsed_sonorus:
            return True
        if self.__forsed_sonorus is False:
            return False
        for son, _ in self.sonorus_deaf_pairs:
            if self.letter == son:
                return True
        return False

    def is_deaf(self):
        """
        Глухая ли согласная.
        """
        if not self.is_consonant():
            return False
        if self.letter in self.forever_deaf:
            return True
        if self.letter in self.forever_sonorus:
            return False
        if self.__forsed_sonorus:
            return False
        if self.__forsed_sonorus is False:
            return True
        for _, df in self.sonorus_deaf_pairs:
            if self.letter == df:
                return True
        return False

    def is_hard(self):
        if not self.is_consonant():
            return False
        if self.letter in self.forever_hard:
            return True
        if self.letter in self.forever_soft:
            return False
        if self.__forced_hard:
            return True
        return False

    def is_soft(self):
        if not self.is_consonant():
            return False
        if self.letter in self.forever_soft:
            return True
        if self.letter in self.forever_hard:
            return False
        if self.__forced_hard is False:
            return True
        return False

    def end(self, string):
        """
        Проверяет, заканчивается ли последовательность букв переданной строкой.
        Скан производится, без учёта текущей.
        """
        prev = self._prev_letter()
        for s in reversed(string):
            if prev.letter != s:
                return False
            if not prev:
                return False
            prev = prev._prev_letter()
        return True

    def is_softener(self, let):
        """
        Является ли символ смягчающим.
        :let: Объект буквы, которую пытаемся смягчить.
        """
        if let.letter in let.forever_hard:
            return False
        if not let.is_consonant():
            return False
        if self.letter in self.vovels_set_soft:
            return True
        if self.letter == 'ь':
            return True
        if self.is_soft() and (let.letter in "дзнст"):
            return True
        if self.letter == 'ъ':
            if self.end("раз") or self.end("из") or self.end("с"):
                return True
        return False

    def is_vowel(self):
        return (self.letter in self.vowels)

    def is_consonant(self):
        return (self.letter in self.consonants)

    def is_mark(self):
        return (self.letter in self.marks)

    def is_shock(self):
        return self.__shock
    
    def classify(self):
        return {
            'paired': self.is_paired_consonant(),
            'sonorus': self.is_sonorus(),
            'deaf': self.is_deaf(),
            'hard': self.is_hard(),
            'soft': self.is_soft(),
            'vowel': self.is_vowel(),
            'consonant': self.is_consonant(),
            'mark': self.is_mark(),
            'shock': self.is_shock()
        }
