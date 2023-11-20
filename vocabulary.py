from collections import Counter
from itertools import chain


class Vocabulary:
    START_TOKEN = "<p>"
    END_TOKEN = "</p>"
    UNK_TOKEN = "<unk>"

    def __init__(self, size=None):
        self.size = size

    def fit(self, tokenized_texts):
        # Creamos una sola lista con todos los tokens
        tokens = chain.from_iterable(tokenized_texts)

        # Contamos las ocurrencias de cada token
        self.single_token_counts = Counter(tokens)
        self.total_number_of_tokens = sum(self.single_token_counts.values())

        # Obtenemos los tokens más comunes con `most_common`, si `size` es `None` entonces se obtienen todos los tokens
        # Si `size` no es `None` entonces se obtienen los `size` tokens más comunes menos 3 (por los tokens especiales)
        top_counts = self.single_token_counts.most_common(
            None if self.size is None else (self.size - 3)
        )

        # Creamos el vocabulario con los tokens especiales y los tokens más comunes
        vocab = [self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN] + [
            w for w, _ in top_counts
        ]

        # Creamos los diccionarios para convertir de id a palabra y viceversa
        self.id_to_word = dict(enumerate(vocab))
        self.word_to_id = {v: k for k, v in iter(self.id_to_word.items())}

        # Establecemos estas variables para acceder más fácilmente a la cantidad de tokens y al tamaño del vocabulario
        self.size = len(self.id_to_word)
        self.tokenset = set(self.word_to_id.keys())

        # Almacenamos los ids de los tokens especiales
        self.start_id = self.word_to_id[self.START_TOKEN]
        self.end_id = self.word_to_id[self.END_TOKEN]
        self.unk_id = self.word_to_id[self.UNK_TOKEN]

    def transform(self, tokenized_texts):
        padded_posts = (
            [self.START_TOKEN, self.START_TOKEN] + p + [self.END_TOKEN]
            for p in tokenized_texts
        )
        cannonical_posts = [[self.word_to_id.get(w, self.unk_id) for w in p] for p in padded_posts]
        return cannonical_posts

    def words_to_ids(self, words):
        return [self.word_to_id.get(w, self.unk_id) for w in words]

    def ids_to_words(self, ids):
        return [self.id_to_word[i] for i in ids]

    def ordered_words(self):
        """Return a list of words, ordered by id."""
        return self.ids_to_words(range(self.size))
