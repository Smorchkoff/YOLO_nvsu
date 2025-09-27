from math import log

text0 = 'синий красный синий фиолетовый'.split()
text1 = 'зеленый красный фиолетовый синий'.split()
text2 = 'желтый зеленый сиреневый сиреневый'.split()
text3 = 'фиолетовый синий сиреневый красный'.split()

docs = [
    text0,
    text1,
    text2,
    text3
]

def terminate_tf(word: str, document: list):
    return document.count(word) / len(document)

def terminate_df(word: str):
    return sum(1 for doc in docs if word in doc)

def terminate_idf(word: str):
    df_word = terminate_df(word)
    return log(len(docs)/df_word)

def terminate_tf_idf(word: str, document: list):
    tf_word = terminate_tf(word, document)
    idf_word = terminate_idf(word)
    return round(tf_word * idf_word, 2)


# NOTE Задание 1 | Посчитайте число уникальных слов в коллекции
unique_words = set()
for doc in docs:
    unique_words.update(doc)
# NOTE Задание 2 | Посчитайтеdf(зеленый)
df_green = terminate_df('зеленый')

#NOTE Задание 3 | Посчитайте f(синий, text0)
blue_count_words_0 = text0.count("синий")

#NOTE Задание 4
# Посчитайте tf-idf(фиолетовый, text3 ).
# Округлять следует до второго знака после запятой, считать, что tf (termfrequency) определяется числом слов в документе.
# Использовать натуральные логарифмы в вычислениях
tf_idf_purple_3 = terminate_tf_idf('фиолетовый', text3)

#NOTE Задание 5
# Посчитайте tf-idf для всех слов в коллекции.
# Округлять следует до второго знака после запятой, считать, что tf (termfrequency) определяется числом слов в документе.
# Использовать натуральные логарифмы в вычислениях.

def task5():
    col_width = 15
    header = " " * col_width + "".join(f"{word:>{col_width}}|" for word in sorted(unique_words))
    print(header)
    tfidf_data = {}
    for i,doc in enumerate(docs):
        tfidf_data[f'text{i}'] = []
        for word in sorted(unique_words):
            tfidf_data[f'text{i}'].append(terminate_tf_idf(word, doc))
    for doc_id in ['text0', 'text1', 'text2', 'text3']:
        row = f"{'|'+doc_id:>{col_width}}" + "".join(f"{val:>{col_width}.2f}|" for val in tfidf_data[doc_id])
        print(row)








if __name__ == '__main__':
    print(f'1)Количество уникальных слов в коллекции: {len(unique_words)}')
    print(f'2)df(зеленый): {df_green}')
    print(f'3)f(синий, text0):{blue_count_words_0}')
    print(f'4)tf_idf(фиолетовый, text3): {tf_idf_purple_3}')
    print(f'5)tf_idf для всех:')
    task5()




