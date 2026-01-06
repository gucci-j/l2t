import random
from nltk.corpus import wordnet

def _get_meaningless_words():
    all_words = []
    for pos in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ]:
        all_words.extend(list(wordnet.all_synsets(pos=pos)))
    meaningless_words = [
        syn.lemmas()[0].name() 
        for syn in random.sample(all_words, 100000)
        if syn.lemmas() and "_" not in syn.lemmas()[0].name()
    ]
    return meaningless_words

# Global variable to store precomputed words
MEANINGLESS_WORDS = _get_meaningless_words()


def _identify_meaningless_word_list(
    text: str
) -> tuple:
    """
    Identify meaningless words in the text.
    
    Args:
        text (str): The input text.
        
    Returns:
        tuple: A tuple containing the instruction and the meaningless words.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Identify the meaningless word(s) in the following text: \n\nThe quick brown fox jumps over the lazy mountain.",
            '["mountain"]'
        )
    """
    instruction_variants = (
        "List all meaningless words in the text below (provide answer as a list):",
        "What are the meaningless words in this text? Please provide them as a list:",
        "Find and list all meaningless words that appear in the following text:",
        "Return a list of all meaningless words from the text below:",
        "Read the text and identify all meaningless words. Format your answer as a list:",
        "Which words in this text are meaningless? Present them in a list format:",
        "Extract all meaningless words from the text and present them as a list:",
        "Analyze the text and list all meaningless words you find:",
        "Please identify and list all meaningless words in the following passage:",
        "Find all out-of-place words in this text and provide them as a list:",
    )
    words = text.split()
    ratio = random.uniform(0.05, 0.10)
    num_words_to_replace = max(1, round(len(words) * ratio))
    indices_to_replace = random.sample(range(len(words)), num_words_to_replace)
    meaningless_words = []
    modified_words = words.copy()
    for idx in indices_to_replace:
        if words[idx][0].isupper():
            modified_word = random.choice(MEANINGLESS_WORDS).capitalize()
        else:
            modified_word = random.choice(MEANINGLESS_WORDS).lower()
        modified_words[idx] = modified_word
        meaningless_words.append(modified_word)
    modified_text = " ".join(modified_words)
    instruction = random.choice(instruction_variants)
    prompt = f"{instruction}\n\n{modified_text}"
    return prompt, str(meaningless_words)


def _identify_meaningless_word_list_with_hint(
    text: str
) -> tuple:
    """
    Identify meaningless words in the text with a hint.
    
    Args:
        text (str): The input text.
        
    Returns:
        tuple: A tuple containing the instruction, the meaningless words, and the hint.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Identify the meaningless word(s) in the following text: \n\nThe quick brown fox jumps over the lazy mountain.\n\nHint: There is 1 meaningless word in the text.",
            "["mountain"]",
        )
    """
    instruction_variants = (
        "Please identify all meaningless words in the text below and provide them as a list:",
        "Find all meaningless words in this text and present your answer as a list:",
        "Review the text and list any meaningless words you find:",
        "What meaningless words appear in this text? Present your answer as a list:",
        "Analyze the following text and identify meaningless words. Format your response as a list:",
        "Spot all meaningless words in the text and provide them in list format:", 
        "Read through this text and list any meaningless words you discover:",
        "Locate all meaningless words in the passage and provide them as a list:",
        "Examine the text and identify meaningless words. Present them in a list format:",
        "Search for meaningless words in this text and list them all:",
    )
    words = text.split()
    ratio = random.uniform(0.05, 0.10)
    num_words_to_replace = max(1, round(len(words) * ratio))
    indices_to_replace = random.sample(range(len(words)), num_words_to_replace)
    meaningless_words = []
    modified_words = words.copy()
    for idx in indices_to_replace:
        if words[idx][0].isupper():
            modified_word = random.choice(MEANINGLESS_WORDS).capitalize()
        else:
            modified_word = random.choice(MEANINGLESS_WORDS).lower()
        modified_words[idx] = modified_word
        meaningless_words.append(modified_word)
    modified_text = " ".join(modified_words)
    instruction = random.choice(instruction_variants)
    hint = f"\n\nHint: There are {len(meaningless_words)} meaningless word(s) in the text."
    prompt = f"{instruction}\n\n{modified_text}{hint}"
    return prompt, str(meaningless_words)


def _generate_correct_word_list(
    text: str
) -> tuple:
    """
    Generate a list of correct words in the text.
    
    Args:
        text (str): The input text.
        
    Returns:
        tuple: A tuple containing the instruction and the correct words.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Identify the meaningless word(s) and provide the correct word(s) in the following text: \n\nThe quick brown fox jumps over the lazy mountain.",
            '["dog"]'
        )
    """
    instruction_variants = (
        "What would be the correct words to replace the meaningless ones in this text? Present your answer as a list:",
        "Some words in this text have been replaced with meaningless ones. What are the original words? Answer in list format:",
        "List the original words that should replace the meaningless substitutions in this text:",
        "Given the text below, what words have been replaced with meaningless ones? Provide the correct words as a list:",
        "This text contains some meaningless substitutions. What were the original words? Present them in a list:",
        "Please identify the original words that were replaced with meaningless ones. Format your answer as a list:",
        "Find the correct words that should replace the meaningless substitutions and provide them as a list:",
        "What are the original words that belong in place of the meaningless ones? Answer with a list:",
        "Determine the correct words that were replaced by meaningless substitutions. Present them in list format:",
        "Identify the original words that were substituted with meaningless ones. Provide your answer as a list:",
    )
    words = text.split()
    ratio = random.uniform(0.05, 0.10)
    num_words_to_replace = max(1, round(len(words) * ratio))
    indices_to_replace = random.sample(range(len(words)), num_words_to_replace)
    correct_words = []
    modified_words = words.copy()
    for idx in indices_to_replace:
        correct_words.append(words[idx])
        if words[idx][0].isupper():
            modified_words[idx] = random.choice(MEANINGLESS_WORDS).capitalize()
        else:
            modified_words[idx] = random.choice(MEANINGLESS_WORDS).lower()
    modified_text = " ".join(modified_words)
    instruction = random.choice(instruction_variants)
    prompt = f"{instruction}\n\n{modified_text}"
    return prompt, str(correct_words)


def _generate_correct_word_list_with_hint(
    text: str
) -> tuple:
    """
    Generate a list of correct words in the text with a hint.
    
    Args:
        text (str): The input text.
        
    Returns:
        tuple: A tuple containing the instruction, the correct words, and the hint.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Identify the meaningless word(s) and provide the correct word(s) in the following text: \n\nThe quick brown fox jumps over the lazy mountain.\n\nHint: There is 1 correct word in the text.",
            '["dog"]',
        )
    """
    instruction_variants = (
        "What would be the correct replacements for the meaningless words in this text? Present your answer as a list:",
        "Some words in this text need to be replaced. What are the original correct words? Answer in list format:",
        "Identify the original words that should replace the meaningless ones. Format your response as a list:", 
        "What were the correct words before they were replaced with meaningless ones? Provide them as a list:",
        "Find the original words that belong in place of the meaningless substitutions. Present as a list:",
        "This text contains modified words. What are their correct replacements? Give your answer as a list:",
        "Can you determine the proper words that were substituted? Present them in list format:",
        "List the original correct words that should appear instead of the meaningless ones:",
        "What words were replaced by meaningless substitutions? Provide the correct ones as a list:",
        "Identify and list the proper words that should replace the meaningless substitutions:",
    )
    words = text.split()
    ratio = random.uniform(0.05, 0.10)
    num_words_to_replace = max(1, round(len(words) * ratio))
    indices_to_replace = random.sample(range(len(words)), num_words_to_replace)
    correct_words = []
    modified_words = words.copy()
    for idx in indices_to_replace:
        correct_words.append(words[idx])
        if words[idx][0].isupper():
            modified_words[idx] = random.choice(MEANINGLESS_WORDS).capitalize()
        else:
            modified_words[idx] = random.choice(MEANINGLESS_WORDS).lower()
    modified_text = " ".join(modified_words)
    instruction = random.choice(instruction_variants)
    hint = f"\n\nHint: There are {len(correct_words)} meaningless word(s) in the text."
    prompt = f"{instruction}\n\n{modified_text}{hint}"
    return prompt, str(correct_words)


def _generate_tuples(
    text: str
) -> tuple:
    """
    Generate tuples of meaningless and correct words in the text.
    
    Args:
        text (str): The input text.
        
    Returns:
        tuple: A tuple containing the instruction and the tuples.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Identify the meaningless word(s) and provide the correct word(s) in the following text: \n\nThe quick brown fox jumps over the lazy mountain.",
            "[(\"mountain\", \"dog\")]"
        )
    """
    instruction_variants = (
        "Find all meaningless words in the text and provide them with their correct replacements as a list of tuples (meaningless_word, correct_word):",
        "For each meaningless word in the text, create a tuple with its correct replacement. Present your answer as a list of (meaningless_word, correct_word) pairs:",
        "Analyze the text and return a list of tuples, where each tuple contains (meaningless_word, proper_word):",
        "Identify all meaningless words and their correct counterparts. Format as [(meaningless, correct), ...]:",
        "Present the meaningless words and their correct replacements as a list of tuples in the format [(wrong_word, right_word), ...]:",
        "Create a list of pairs showing each meaningless word and its correct replacement in tuple format (meaningless, correct):",
        "Match each meaningless word with its correct replacement and present as tuples in the format [(meaningless, correct), ...]:",
        "List all meaningless-correct word pairs as tuples in the format [(meaningless_word, correct_word), ...]:",
        "Provide pairs of meaningless words and their correct replacements as [(wrong_word, right_word), ...]:",
        "Return a list of tuples where each tuple contains a meaningless word and its proper replacement (meaningless, correct):",
    )
    words = text.split()
    ratio = random.uniform(0.05, 0.10)
    num_words_to_replace = max(1, round(len(words) * ratio))
    indices_to_replace = random.sample(range(len(words)), num_words_to_replace)
    tuples = []
    modified_words = words.copy()
    for idx in indices_to_replace:
        correct_word = words[idx]
        meaningless_word = ""
        if words[idx][0].isupper():
            meaningless_word = random.choice(MEANINGLESS_WORDS).capitalize()
        else:
            meaningless_word = random.choice(MEANINGLESS_WORDS).lower()
        modified_words[idx] = meaningless_word
        tuples.append((meaningless_word, correct_word))
    modified_text = " ".join(modified_words)
    instruction = random.choice(instruction_variants)
    prompt = f"{instruction}\n\n{modified_text}"
    return prompt, str(tuples)


def _predict_number_of_replaced_words(
    text: str
) -> tuple:
    """
    Predict the number of replaced words in the text.
    
    Args:
        text (str): The input text.
        
    Returns:
        tuple: A tuple containing the instruction and the number of replaced words.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Predict the number of meaningless words in the following text: \n\nThe quick brown fox jumps over the lazy mountain.",
            "1"
        )
    """
    instruction_variants = (
        "How many meaningless words appear in this text?",
        "Count the number of meaningless words in the following passage:",
        "In this text, what is the total count of meaningless words?",
        "Determine the total number of meaningless words present in this text:",
        "Calculate how many meaningless words are in the following text:",
        "What's the count of meaningless words in this passage?",
        "Tell me the number of meaningless words you can find in this text:",
        "How many words in this text would you classify as meaningless?",
        "Please count the meaningless words in the following text:",
        "Identify the total number of meaningless words in this passage:",
    )
    words = text.split()
    ratio = random.uniform(0.05, 0.10)
    num_words_to_replace = max(1, round(len(words) * ratio))
    indices_to_replace = random.sample(range(len(words)), num_words_to_replace)
    modified_words = words.copy()
    for idx in indices_to_replace:
        if words[idx][0].isupper():
            modified_words[idx] = random.choice(MEANINGLESS_WORDS).capitalize()
        else:
            modified_words[idx] = random.choice(MEANINGLESS_WORDS).lower()
    modified_text = " ".join(modified_words)
    instruction = random.choice(instruction_variants)
    prompt = f"{instruction}\n\n{modified_text}"
    return prompt, str(num_words_to_replace)


def _predict_number_of_replaced_words_with_reasoning(
    text: str
) -> tuple:
    """
    Predict the number of replaced words in the text with reasoning.
    
    Args:
        text (str): The input text.
        
    Returns:
        tuple: A tuple containing the instruction and the reasoning.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Based on the context, how many words in this text appear to be meaningless? Support your answer with reasoning: \n\nThe quick brown fox jumps over the lazy mountain.",
            "Based on the context, the word(s) 'mountain' does not fit the passage. Hence, the total number of meaningless words is 1."
        )
    """
    # More natural instruction variants with explicit request for reasoning
    instruction_variants = (
        "Looking at this text, how many words seem out of place or meaningless? Explain your reasoning.",
        "Can you analyze this passage and count any words that don't fit the context? Please provide your reasoning.", 
        "Read through this text and tell me how many meaningless words you spot. Explain how you determined this.",
        "Based on the context, how many words in this text appear to be meaningless? Support your answer with reasoning.",
        "Taking a careful look at this passage, count the number of words that don't make sense and explain why.",
        "Examine this text and determine how many words are contextually meaningless. Provide your reasoning.",
        "Could you identify the number of words that seem inappropriate in this text? Explain your thought process.",
        "How many words in this passage appear to be used incorrectly? Please justify your answer.",
        "Count the number of words that break the logical flow of this text and explain your analysis.",
        "In your analysis, how many meaningless words did you find in this passage? Provide your reasoning."
    )
    words = text.split()
    ratio = random.uniform(0.05, 0.10)
    num_words_to_replace = max(1, round(len(words) * ratio))
    indices_to_replace = random.sample(range(len(words)), num_words_to_replace)
    modified_words = words.copy()
    for idx in indices_to_replace:
        if words[idx][0].isupper():
            modified_words[idx] = random.choice(MEANINGLESS_WORDS).capitalize()
        else:
            modified_words[idx] = random.choice(MEANINGLESS_WORDS).lower()
    modified_text = " ".join(modified_words)
    instruction = random.choice(instruction_variants)
    prompt = f"{instruction}\n\n{modified_text}"
    reasoning_templates = [
        f"Based on the context, the word(s) '{', '.join(modified_words[idx] for idx in indices_to_replace)}' do not fit the passage. Hence, the total number of meaningless words is {num_words_to_replace}.",
        f"After analyzing the text, I found that the word(s) '{', '.join(modified_words[idx] for idx in indices_to_replace)}' do not make sense in the context. Therefore, the total number of meaningless words is {num_words_to_replace}.",
        f"I have identified the word(s) '{', '.join(modified_words[idx] for idx in indices_to_replace)}' as meaningless in the passage. This leads to a total of {num_words_to_replace} meaningless words in the text.",
        f"Upon reviewing the text, I noticed that the word(s) '{', '.join(modified_words[idx] for idx in indices_to_replace)}' are out of place. Consequently, the total number of meaningless words is {num_words_to_replace}.",
        f"By examining the text, I determined that the word(s) '{', '.join(modified_words[idx] for idx in indices_to_replace)}' are inappropriate. This results in a total of {num_words_to_replace} meaningless words in the passage.",
        f"Having analyzed the text, I found that the word(s) '{', '.join(modified_words[idx] for idx in indices_to_replace)}' are meaningless. Therefore, the total number of meaningless words is {num_words_to_replace}.",
        f"Upon careful examination, I discovered that the word(s) '{', '.join(modified_words[idx] for idx in indices_to_replace)}' are incorrect in the context. This leads to a total of {num_words_to_replace} meaningless words in the text.",
        f"After reviewing the text, I identified the word(s) '{', '.join(modified_words[idx] for idx in indices_to_replace)}' as meaningless. Hence, the total number of meaningless words is {num_words_to_replace}.",
        f"By analyzing the text, I found that the word(s) '{', '.join(modified_words[idx] for idx in indices_to_replace)}' are out of place. This results in a total of {num_words_to_replace} meaningless words in the passage.",
        f"Based on my analysis, the word(s) '{', '.join(modified_words[idx] for idx in indices_to_replace)}' do not fit the context. Therefore, the total number of meaningless words is {num_words_to_replace}.",
    ]
    reasoning = random.choice(reasoning_templates)
    return prompt, reasoning
    

def _generate_tuples_with_hint(
    text: str
) -> tuple:
    """
    Generate tuples of meaningless and correct words in the text with a hint.
    
    Args:
        text (str): The input text.
        
    Returns:
        tuple: A tuple containing the instruction, the tuples, and the hint.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Identify the meaningless word(s) and provide the correct word(s) in the following text: \n\nThe quick brown fox jumps over the lazy mountain.\n\nHint: There is 1 tuple in the text.",
            "[('mountain', 'dog')]",
        )
    """
    instruction_variants = (
        "Find the meaningless-correct word pairs in this text and present them as tuples. Format: [(meaningless, correct), ...]:",
        "For each out-of-place word, create a tuple with its proper replacement. Show as [(wrong_word, right_word), ...]:", 
        "Match each meaningless word with what it should be. Present as [(meaningless, correct), ...]:",
        "List pairs of misused words and their correct versions as [(incorrect, correct), ...]:",
        "Create tuples showing each meaningless word and what word it should be. Format: [(meaningless, proper), ...]:",
        "Identify word pairs where meaningless words need replacement. Show as [(meaningless, correct), ...]:",
        "Point out the meaningless-correct word connections using tuples: [(wrong, right), ...]:",
        "Show which words are meaningless and what they should be. Format: [(meaningless, correct), ...]:",
        "Link each meaningless word to its correct replacement using tuples: [(incorrect, correct), ...]:",
        "Pair up the meaningless words with their proper alternatives as [(meaningless, correct), ...]:"
    )
    words = text.split()
    ratio = random.uniform(0.05, 0.10)
    num_words_to_replace = max(1, round(len(words) * ratio))
    indices_to_replace = random.sample(range(len(words)), num_words_to_replace)
    tuples = []
    modified_words = words.copy()
    for idx in indices_to_replace:
        correct_word = words[idx]
        meaningless_word = ""
        if words[idx][0].isupper():
            meaningless_word = random.choice(MEANINGLESS_WORDS).capitalize()
        else:
            meaningless_word = random.choice(MEANINGLESS_WORDS).lower()
        modified_words[idx] = meaningless_word
        tuples.append((meaningless_word, correct_word))
    modified_text = " ".join(modified_words)
    instruction = random.choice(instruction_variants)
    hint = f"\n\nHint: There are {len(tuples)} meaningless word(s) in the text."
    prompt = f"{instruction}\n\n{modified_text}{hint}"
    return prompt, str(tuples)


def _generate_corrected_text(
    text: str,
    no_instruction_variation: bool = False,
    no_instruction: bool = False,
    **kwargs
) -> tuple:
    """
    Generate the corrected text with meaningless words.
    
    Args:
        text (str): The input text.
        no_instruction_variation (bool): A flag to disable instruction variation.
        no_instruction (bool): A flag to disable instruction.
        
    Returns:
        tuple: A tuple containing the instruction and the corrected text.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Identify the meaningless word(s) and generate the corrected text: \n\nThe quick brown fox jumps over the lazy mountain.",
            "The quick brown fox jumps over the lazy dog."
        )
    """
    instruction_variants = (
        "Fix this text by replacing meaningless words.",
        "Correct the text by replacing any meaningless words.",
        "Rewrite this text, replacing meaningless words with appropriate ones.",
        "Please fix the meaningless words in this text.",
        "Replace the meaningless words to correct this text.",
        "Edit this text to remove any meaningless words.",
        "Revise this passage by fixing the meaningless words.",
        "Correct any meaningless words in the following text.",
        "Improve this text by replacing the meaningless words.",
        "Rewrite this passage with the correct words instead of meaningless ones.",
        "Fix the meaningless words in this passage.",
        "Replace nonsensical words with appropriate ones.",
        "Correct this text by fixing all meaningless words.",
        "Substitute proper words for the meaningless ones.",
        "Make this text coherent by fixing meaningless words.",
        "Identify and fix all meaningless words here.",
        "Clean up this text by replacing meaningless terms.",
        "Restore meaning by fixing the inappropriate words.",
        "Correct all out-of-place words in this text.",
        "Find and replace the meaningless words.",
    )
    words = text.split()
    ratio = random.uniform(0.05, 0.10)
    num_words_to_replace = max(1, round(len(words) * ratio))
    indices_to_replace = random.sample(range(len(words)), num_words_to_replace)
    correct_words = []
    modified_words = words.copy()
    for idx in indices_to_replace:
        correct_words.append(words[idx])
        if words[idx][0].isupper():
            modified_words[idx] = random.choice(MEANINGLESS_WORDS).capitalize()
        else:
            modified_words[idx] = random.choice(MEANINGLESS_WORDS).lower()
    modified_text = " ".join(modified_words)
    if no_instruction:
        return modified_text, " ".join(text.split())
    if no_instruction_variation:
        instruction = "Fix this text by replacing meaningless words."
    else:
        instruction = random.choice(instruction_variants)
    # 50% chance to swap the instruction and context
    if random.random() < 0.5:
        instruction, modified_text = modified_text, instruction
    prompt = f"{instruction}\n\n{modified_text}"
    return prompt, " ".join(text.split())


def _generate_corrected_text_with_hint(
    text: str
) -> tuple:
    """
    Generate the corrected text with meaningless words and a hint.
    
    Args:
        text (str): The input text.
        
    Returns:
        tuple: A tuple containing the instruction, the corrected text, and the hint.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Identify the meaningless word(s) and generate the corrected text: \n\nThe quick brown fox jumps over the lazy mountain.\n\nHint: There are 1 meaningless word(s) in the text.",
            "The quick brown fox jumps over the lazy dog.",
        )
    """
    instruction_variants = (
        "Please restore this text to its original form by replacing the meaningless words:",
        "This text contains some misplaced words. Can you write out how it should correctly read?",
        "Rewrite this passage, fixing any words that don't belong in the context:",
        "Some words in this text need to be corrected. What's the proper version?",
        "Could you provide the original version of this text without the meaningless substitutions?",
        "This passage has been altered with inappropriate words. How should it actually read?",
        "Please reconstruct this text by replacing the out-of-place words with proper ones:",
        "What would be the correct version of this text when all meaningless words are fixed?",
        "Revise this text by replacing any meaningless words with their appropriate alternatives:",
        "After correcting the meaningless words, how should this passage properly read?"
    )
    words = text.split()
    ratio = random.uniform(0.05, 0.10)
    num_words_to_replace = max(1, round(len(words) * ratio))
    indices_to_replace = random.sample(range(len(words)), num_words_to_replace)
    correct_words = []
    modified_words = words.copy()
    for idx in indices_to_replace:
        correct_words.append(words[idx])
        if words[idx][0].isupper():
            modified_words[idx] = random.choice(MEANINGLESS_WORDS).capitalize()
        else:
            modified_words[idx] = random.choice(MEANINGLESS_WORDS).lower()
    modified_text = " ".join(modified_words)
    instruction = random.choice(instruction_variants)
    hint = f"\n\nHint: There are {len(correct_words)} meaningless word(s) in the text."
    prompt = f"{instruction}\n\n{modified_text}{hint}"
    return prompt, " ".join(text.split())


def generate_random_word_instruction_data(
    example: dict,
    batched: bool = False,
    min_num_words: int = 5,
    use_only_primary: bool = False,
    **kwargs
) -> dict:
    """
    Generate instruction data for random word identification task.
    """
    task_funcs = {
        "identify_meaningless_word_list": _identify_meaningless_word_list,
        "identify_meaningless_word_list_with_hint": _identify_meaningless_word_list_with_hint,
        "generate_correct_word_list": _generate_correct_word_list,
        "generate_correct_word_list_with_hint": _generate_correct_word_list_with_hint,
        "generate_tuples": _generate_tuples,
        "predict_number_of_replaced_words": _predict_number_of_replaced_words,
        "predict_number_of_replaced_words_with_reasoning": _predict_number_of_replaced_words_with_reasoning,
        "generate_tuples_with_hint": _generate_tuples_with_hint,
        "generate_corrected_text": _generate_corrected_text,
        "generate_corrected_text_with_hint": _generate_corrected_text_with_hint,
    }
    task_variants = list(task_funcs.keys())
    
    if batched:
        if use_only_primary:
            task_variant = "generate_corrected_text"
            results = [
                task_funcs[task_variant](text, kwargs.get("no_instruction_variation", False), kwargs.get("no_instruction", False))
                for text in example["text"]
                if len(text.split()) >= min_num_words
            ]
        else:
            results = [
                task_funcs[random.choice(task_variants)](text)
                for text in example["text"]
                if len(text.split()) >= min_num_words
            ]
        if results:
            prompts, completions = zip(*results)
        else:
            prompts, completions = [], []
        return {"prompt": list(prompts), "completion": list(completions)}
    else:
        if use_only_primary:
            task = "generate_corrected_text"
        else:
            task = random.choice(task_variants)
        text = example["text"]
        prompt, completion = task_funcs[task](text)
        return {"prompt": prompt, "completion": completion}
