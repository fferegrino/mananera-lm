from typing import Tuple

import numpy as np

from add_k_trigram_lm import AddKTrigramLM
from vocabulary import Vocabulary


class SequenceGenerator:
    def __init__(self, language_model: AddKTrigramLM, vocabulary: Vocabulary):
        self.lm = language_model
        self.vocab = vocabulary

    def sample_next_token(self, *sequence: Tuple[str]) -> str:
        # Esto busca cada palabra en el vocabulario y obtiene su probabilidad condicional.
        # Esto puede ser lento si el vocabulario es muy grande, ¿podríamos hacerlo mejor?
        probs = [self.lm.next_token_proba(word, sequence) for word in self.lm.tokens]

        # Elegimos una palabra al azar de acuerdo a sus probabilidades
        return np.random.choice(self.lm.tokens, p=probs)

    def generate_sequences(self, *start: Tuple[str], max_length: int = 200) -> str:
        # Given it the start sequence to indicate the start of a post.
        seq = [self.vocab.start_id, self.vocab.start_id]
        if start:
            start = [self.vocab.word_to_id[w] for w in start]
            seq.extend(start)
        for i in range(max_length):
            seq.append(self.sample_next_token(*seq))
            # Stop at post
            if seq[-1] == self.vocab.end_id:
                break
        return " ".join([f"{self.vocab.id_to_word[s]} ({s})" for s in seq])
