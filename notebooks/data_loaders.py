import pandas as pd
from Phonetic import isEnglish
from feature_engineering import *

def get_faculty_members(filename):
    """
    Loads faculty members as strings from given file
    :param filename: path to file
    :return: lines of names
    """
    members = []
    with open (filename, 'r') as fp:
        members = fp.read()
    return members.split('\n')


def load_faculty_data (
        path_gr = '../input/griffindor.txt', 
        path_hf = '../input/hufflpuff.txt',
        path_rv = '../input/ravenclaw.txt',
        path_sl = '../input/slitherin.txt'):
    """
    Loads all the available raw data to variables as sequence of lines.
    :param path_gr: path to griffindor file
    :param path_hf: path to hufflpuff file
    :param path_rv: path to ravenclaw file
    :param path_sl: path to slitherin file
    :return: (griffindor, hufflpuff, ravenclaw, slitherin)
    """
    
    griffindor = get_faculty_members(path_gr)
    hufflpuff = get_faculty_members(path_hf)
    ravenclaw = get_faculty_members(path_rv)
    slitherin = get_faculty_members(path_sl)
    
    return griffindor, hufflpuff, ravenclaw, slitherin


def parse_line_to_hogwarts_df (line):
    """
    Parses single input line to dataset element.
    :param line: input line
    :return: named dict of features
    """
    splitted_line = line.split(' ')
    name = splitted_line[0]
    name_count = count_letter_type(name)
    surname = " ".join(splitted_line[1:])
    surname_count = count_letter_type(surname)
    return {
     'name': name, 
     'surname': surname, 
     'is_english': isEnglish(line),
     'name_starts_with_vowel': starts_with_letter(name), 
     'name_starts_with_consonant': starts_with_letter(name, 'consonant'),
     'name_ends_with_vowel': ends_with_letter(name), 
     'name_ends_with_consonant': ends_with_letter(name, 'consonant'),
     'name_length': len(name), 
     'name_vowels_count': name_count['vowel'],
     'name_double_vowels_count': count_double_letter(name),
     'name_consonant_count': name_count['consonant'],
     'name_double_consonant_count': count_double_letter(name, 'consonant'),
     'name_paired_count': name_count['paired'],
     'name_deaf_count': name_count['deaf'],
     'name_sonorus_count': name_count['sonorus'],
     'surname_starts_with_vowel': starts_with_letter(surname), 
     'surname_starts_with_consonant': starts_with_letter(surname, 'consonant'),
     'surname_ends_with_vowel': ends_with_letter(surname), 
     'surname_ends_with_consonant': ends_with_letter(surname, 'consonant'),
     'surname_length': len(surname), 
     'surname_vowels_count': surname_count['vowel'],
     'surname_double_vowels_count': count_double_letter(surname),
     'surname_consonant_count': surname_count['consonant'],
     'surname_double_consonant_count': count_double_letter(surname, 'consonant'),
     'surname_paired_count': surname_count['paired'],
     'surname_deaf_count': surname_count['deaf'],
     'surname_sonorus_count': surname_count['sonorus'],
    }


def load_processed_data ():
    """
    Loads data and gets features for each element.
    :return: pandas.DataFrame of data
    """

    griffindor, hufflpuff, ravenclaw, slitherin = load_faculty_data()
    
    persons = []
    for person in griffindor:
        featurized_person = parse_line_to_hogwarts_df(person)
        featurized_person['is_griffindor'] = 1
        featurized_person['is_hufflpuff'] = 0
        featurized_person['is_ravenclaw'] = 0
        featurized_person['is_slitherin'] = 0
        persons.append(featurized_person)
    for person in hufflpuff:
        featurized_person = parse_line_to_hogwarts_df(person)
        featurized_person['is_griffindor'] = 0
        featurized_person['is_hufflpuff'] = 1
        featurized_person['is_ravenclaw'] = 0
        featurized_person['is_slitherin'] = 0
        persons.append(featurized_person)
    for person in ravenclaw:
        featurized_person = parse_line_to_hogwarts_df(person)
        featurized_person['is_griffindor'] = 0
        featurized_person['is_hufflpuff'] = 0
        featurized_person['is_ravenclaw'] = 1
        featurized_person['is_slitherin'] = 0
        persons.append(featurized_person)
    for person in slitherin:
        featurized_person = parse_line_to_hogwarts_df(person)
        featurized_person['is_griffindor'] = 0
        featurized_person['is_hufflpuff'] = 0
        featurized_person['is_ravenclaw'] = 0
        featurized_person['is_slitherin'] = 1
        persons.append(featurized_person)
    

    hogwarts_df = pd.DataFrame(persons,
        columns=[
         'name', 
         'surname', 
         'is_english',
         'name_starts_with_vowel', 
         'name_starts_with_consonant',
         'name_ends_with_vowel', 
         'name_ends_with_consonant',
         'name_length', 
         'name_vowels_count',
         'name_double_vowels_count',
         'name_consonant_count',
         'name_double_consonant_count',
         'name_paired_count',
         'name_deaf_count',
         'name_sonorus_count',
         'surname_starts_with_vowel', 
         'surname_starts_with_consonant',
         'surname_ends_with_vowel', 
         'surname_ends_with_consonant',
         'surname_length', 
         'surname_vowels_count',
         'surname_double_vowels_count',
         'surname_consonant_count',
         'surname_double_consonant_count',
         'surname_paired_count',
         'surname_deaf_count',
         'surname_sonorus_count',
         'is_griffindor',
         'is_hufflpuff',
         'is_ravenclaw',
         'is_slitherin'
        ]
    )
    
    return hogwarts_df

