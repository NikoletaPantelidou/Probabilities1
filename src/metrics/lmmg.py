from tqdm import tqdm


def get_metalinguistic_prompt_paired(sentence1: str, sentence2: str) -> str:
    prompt = f"""
Which of the following two sentences is more grammatically correct in English? Respond with 1 or 2 as your answer.
Sentence 1: {sentence1}
Sentence 2: {sentence2}
Answer:"""
    return prompt


def evaluate(dataset, model, condition="paired", **kwargs):
    # Initialize storage for results.
    data = []

    # Fetch items from dataset.
    stimuli = dataset.items
    meta_vars = list(stimuli)

    # Evaluate
    for _, example in tqdm(stimuli.iterrows(), total=dataset.n_items):
        # Initialize meta information for this item
        res = {v: example[v] for v in meta_vars}
        res["condition"] = condition

        # Get minimal pair.
        sentence1 = example["sentence1"] + "."  # Change names
        sentence2 = example["sentence2"] + "."  # Change names

        ########################################################################
        # Direct method: minimal pair comparison
        ########################################################################

        sentences = [sentence1, sentence2]  # Change names
        sentence_probs = model.get_logprob_of_sequence(
            sentences,
            reduction="sum"
        )
        res["direct_sum_logprob_sentence1"] = sentence_probs[0]  # Change name
        res["direct_sum_logprob_sentence2"] = sentence_probs[1]  # Change name
        res["direct_correct"] = sentence_probs[0] > sentence_probs[1]

        ########################################################################
        # Metalinguistic method
        ########################################################################
        # Change: the study only analysis the direct method, hence this section has been removed.
        # Store all the results from this item.
        data.append(res)

    return data