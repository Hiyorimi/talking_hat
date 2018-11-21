import pandas as pd
from data_loaders import parse_line_to_hogwarts_df

def get_single_student_features (name):
    """
    Gets features for single object.
    :param name: string representing full name
    :return: pd.DataFrame object with features for the input object
    """
    featurized_person_df = parse_line_to_hogwarts_df(name)


    person_df = pd.DataFrame(featurized_person_df,
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
        ],
                             index=[0]
    )
    featurized_person = person_df.drop(
                        ['name', 'surname'], axis = 1
                        )
    return featurized_person


def get_predictions_vector (model, person):
    """
    Returns result of the model prediction for the input string.
    :param model: trained model
    :param person: string with full name
    :return: list of class probabilites
    """
    encoded_person = get_single_student_features(person)
    return model.predict_proba(encoded_person)[0]


def get_predctions_vector(models, person):
    """
    Gets prediction as the probability of being assigned to each faculty.
    :param models: input models
    :param person: string with full name
    :return: dict with faculties as keys and probabilities as values
    """
    predictions = [get_predictions_vector(model, person)[1] for model in models]
    return {
        'slitherin': predictions[0],
        'griffindor': predictions[1],
        'ravenclaw': predictions[2],
        'hufflpuff': predictions[3]
    }


def score_testing_dataset(models):
    """
    Scores models against pre-defined names
    :param models:
    :return: pd.DataFrame with probablities for each faculty
    """
    testing_dataset = [
        "Кирилл Малев", "Kirill Malev",
        "Гарри Поттер", "Harry Potter",
        "Северус Снейп", "Северус Снегг", "Severus Snape",
        "Том Реддл", "Tom Riddle",
        "Салазар Слизерин", "Salazar Slytherin"]

    data = []
    for name in testing_dataset:
        predictions = get_predctions_vector(models, name)
        predictions['name'] = name
        data.append(predictions)

    scoring_df = pd.DataFrame(data,
                              columns=['name',
                                       'slitherin',
                                       'griffindor',
                                       'hufflpuff',
                                       'ravenclaw'])
    return scoring_df
