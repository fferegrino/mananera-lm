from collections import defaultdict


class AddKTrigramLM:
    """Trigram LM with add-k smoothing."""

    order_n = 3

    def __init__(self, k=0.0):
        self.k = k

    def fit(self, corpus):
        """Entrena el modelo de lenguaje a partir de un corpus."""

        # Creamos un diccionario de diccionarios para guardar las cuentas de los trigramas
        # En este diccionario las ocurrencias del token `w` dado `w_1 w_2` se almacenan como `counts[(w_2,w_1)][w]`
        counts = defaultdict(lambda: defaultdict(lambda: 0))

        # Necesitamos tener una lista de todos los tokens en el corpus
        # Esto es necesario para calcular el tamaño del vocabulario
        # y para calcular las probabilidades de los tokens
        all_tokens = set()

        # Iteramos sobre cada documento del corpus
        # Para cada documento iteramos sobre cada token
        # En cada iteración actualizamos el contexto y las cuentas de los trigramas
        w_1, w_2 = None, None
        for document in corpus:
            for token in document:
                all_tokens.add(token)
                if w_1 is not None and w_2 is not None:
                    counts[(w_2, w_1)][token] += 1
                # Update context
                w_2 = w_1
                w_1 = token

        # En `_context_totals` almacenamos las ocurrencias de los bigramas `w_2,w_1`
        self._context_totals = dict()
        for context, ctr in counts.items():
            self._context_totals[context] = sum(ctr.values())

        # Convertir los defaultdicts en diccionarios normales
        # esto es solo por fines de presentación
        self.totals_ = dict()
        for context, ctr in counts.items():
            self.totals_[context] = dict(ctr)

        self.tokens_ = list(all_tokens)
        self.V_ = len(self.tokens_)




    def set_k(self, k=0.0):
        self.k = k

    def next_token_proba(self, token, current_sequence):
        """Calcula la probabilidad de un token dada una secuencia de tokens."""

        # Obtenemos los 2 últimos tokens de la secuencia: `w_1` y `w_2`
        context = tuple(current_sequence[-2:])

        # Count word es la cuenta de las veces que ocurren los tokens `w_1`, `w_2` y `token` en el corpus
        count_word = self.totals_.get(context, {}).get(token, 0)

        # Context count es la cuenta de las veces que ocurren los tokens `w_1` y `w_2` en el corpus
        context_count = self._context_totals.get(context, 0)


        if self.k == 0:
            # Si k = 0, entonces no se aplica suavizado y la probabilidad es la división de la cuenta de los
            # tokens `w_1`, `w_2` y `token` entre la cuenta de los tokens `w_1` y `w_2`
            return count_word / context_count
        else:
            # El cáculo de la probabilidad es la división de la cuenta de los tokens `w_1`, `w_2` y `token` entre la
            # cuenta de los tokens `w_1` y `w_2`
            return (count_word+self.k) / (context_count + self.k * self.V_)
