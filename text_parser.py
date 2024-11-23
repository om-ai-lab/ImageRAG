from keyphrase_vectorizers import KeyphraseCountVectorizer
import pdb
import spacy
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import nltk


def extract_key_phrases(rewrite_query, original_query, kw_model):
    key_phrases = kw_model.extract_keywords(
        keyphrase_ngram_range=(2, 4),
        docs=rewrite_query,
        top_n=10,
        use_mmr=True,
        diversity=0.7
    )
    # key_phrases = kw_model.extract_keywords(
    #     docs=query,
    #     keyphrase_ngram_range=(2, 4),
    #     use_maxsum=True,
    #     nr_candidates=30,
    #     top_n=5
    # )[::-1]
    print("key phrases: {}".format(key_phrases))

    key_words = kw_model.extract_keywords(
        docs=original_query,
        vectorizer=KeyphraseCountVectorizer(),
        top_n=10,
    )
    print("key words: {}".format(key_words))

    key_intersection = list(set(key_phrases) & set(key_words))

    if len(key_intersection) != 0:
        key_words_nonintersection = [key_word for key_word in key_words if key_word not in key_intersection]
        print("intersection: {}".format(key_intersection))
        key_text = key_intersection + key_words_nonintersection
    else:
        key_text = key_words
        print("no intersection")

    key_text = key_text[:5]
    key_text = [key[0] for key in key_text]
    return key_text


def ner(text):

    # nlp = spacy.load('en_core_web_sm')
    # text_change = text
    # doc = nlp(text_change)
    # entity_list = []
    # for entity in doc.ents:
    #     if entity.label_ == 'GPE':
    #         entity_text = entity.text
    #         entity_list.append(entity_text)

    # st = StanfordNERTagger(
    #     '/media/zilun/wd-161/RS5M/RS5M_v4/nips_rebuttal/geometa/stanford-ner-4.2.0/stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz',
    #     '/media/zilun/wd-161/RS5M/RS5M_v4/nips_rebuttal/geometa/stanford-ner-4.2.0/stanford-ner-2020-11-17/stanford-ner-4.2.0.jar',
    #     encoding='utf-8'
    # )
    # tokenized_text = word_tokenize(text)
    # classified_text = st.tag(tokenized_text)
    # entity_list = []
    # for entity in classified_text:
    #     if entity[1] == 'LOCATION':
    #         entity_text = entity[0]
    #         entity_list.append(entity_text)

    # pos_tagged_sent = nltk.pos_tag(nltk.tokenize.word_tokenize(text))
    # entity_list = [tag[0] for tag in pos_tagged_sent if tag[1] == 'NN']

    import yake
    kw_extractor = yake.KeywordExtractor()
    key_phrases = kw_extractor.extract_keywords(text)
    for kw in key_phrases:
        print(kw)

    # from monkeylearn import MonkeyLearn
    # ml = MonkeyLearn('your_api_key')
    # my_text = """
    # When it comes to evaluating the performance of keyword extractors, you can use some of the standard metrics in machine learning: accuracy, precision, recall, and F1 score. However, these metrics don’t reflect partial matches; they only consider the perfect match between an extracted segment and the correct prediction for that tag.
    # """
    # data = [my_text]
    # model_id = 'your_model_id'
    # result = ml.extractors.extract(model_id, data)
    # dataDict = result.body
    # for item in dataDict[0]['extractions'][:10]:
    #     print(item['parsed_value'])

    # Not Bad
    import spacy
    import pytextrank
    # example text
    # load a spaCy model, depending on language, scale, etc.
    nlp = spacy.load("en_core_web_sm")
    # add PyTextRank to the spaCy pipeline
    nlp.add_pipe("textrank")
    doc = nlp(text)
    # examine the top-ranked phrases in the document
    for phrase in doc._.phrases[:10]:
        print(phrase.text)

    # from collections import Counter
    # from string import punctuation
    # nlp = spacy.load("en_core_web_sm")
    # def get_hotwords(text):
    #     result = []
    #     pos_tag = ['PROPN', 'ADJ', 'NOUN']
    #     doc = nlp(text.lower())
    #     for token in doc:
    #         if (token.text in nlp.Defaults.stop_words or token.text in punctuation):
    #             continue
    #         if (token.pos_ in pos_tag):
    #             result.append(token.text)
    #     return result
    # output = set(get_hotwords(text))
    # most_common_list = Counter(output).most_common(10)
    # for item in most_common_list:
    #     print(item[0])

    ## Not Bad
    # from keybert import KeyBERT
    # kw_model = KeyBERT()
    # key_phrases = kw_model.extract_keywords(
    #     docs=text,
    #     keyphrase_ngram_range=(2, 4),
    #     use_maxsum=True,
    #     nr_candidates=30,
    #     top_n=5
    # )[::-1]
    #
    # key_phrases = kw_model.extract_keywords(
    #     docs=text,
    #     vectorizer=KeyphraseCountVectorizer(),
    #     top_n=10,
    # )

    print(key_phrases)

    return


def main():
    cap = "Suppose the top of this image represents north. How many aircrafts are heading northeast? What is the color of the building rooftop to their southeast?"
    entity_list = ner(cap)
    print()


if __name__ == "__main__":
    main()