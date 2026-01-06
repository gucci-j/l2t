import random
import blingfire as bf
import itertools

_stored_sample = None

from .char_identification import generate_char_identification_instruction_data
from .shuffle import generate_shuffle_instruction_data
from .mlm import generate_mlm_instruction_data
from .random import generate_random_word_instruction_data
from .token_type import generate_token_type_instruction_data
from .half_completion import generate_half_completion_instruction_data
from .typo_correction import generate_typo_correction_instruction_data
from .sentence_reordering import generate_sentence_reordering_instruction_data
from .sentence_deletion import generate_sentence_deletion_instruction_data
from .last_phrase_prediction import generate_last_phrase_prediction_instruction_data
from .passage_completion import generate_passage_completion_instruction_data
from .one_word_completion import generate_one_word_completion_instruction_data
from .char_count import generate_char_count_instruction_data
from .space_insertion import generate_space_insertion_instruction_data


def get_default_task_type(
    example, 
    batched, 
    prev_example, 
    use_only_generation, 
):
    """Return a default task type if none is given."""
    if batched:
        n = len(example["text"])
        candidates = [
            "shuffle", 
            "mlm", 
            "char_identification",
            "random", 
            "typo_correction", 
            "half_completion",
            "one_word_completion",
            "space_insertion",
        ]
        if n >= 3:
            if use_only_generation:
                candidates.extend([
                    "passage_completion", 
                    "char_count", 
                    "token_type", 
                    "sentence_reordering"
                ])
                if prev_example is not None:
                    candidates.extend(
                        ["last_phrase_prediction", "sentence_deletion"]
                    )
            elif prev_example is not None:
                candidates.extend([
                    "token_type", 
                    "sentence_reordering",
                    "char_count",
                    "sentence_deletion",
                    "last_phrase_prediction", 
                ])
            else:
                candidates.extend([
                    "token_type", 
                    "sentence_reordering", 
                    "char_count",
                ])
        elif n == 2:
            if use_only_generation:
                candidates.extend(["char_count", "token_type"])
                if prev_example is not None:
                    candidates.extend([
                        "last_phrase_prediction", "sentence_deletion"
                    ])
            elif prev_example is not None:
                candidates.extend([
                    "token_type",
                    "sentence_deletion", 
                    "last_phrase_prediction", 
                    "char_count",
                ])
            else:
                candidates.extend([
                    "token_type", "char_count",
                ])
        else:
            if use_only_generation:
                candidates.extend(["char_count", "token_type"])
            else:
                candidates.extend([
                    "token_type", "char_count",
                ])
    else:
        candidates = [
            "shuffle", "mlm", "char_identification", "random",
            "half_completion", "typo_correction", "one_word_completion",
            "space_insertion"
        ]
        if use_only_generation:
            candidates.extend(["char_count", "token_type"])
        else:
            candidates.extend([
                "token_type", "char_count"
            ])
    return random.choice(candidates)


# Chunking functions for batched tasks that need special processing.
def chunk_text_as_str_by_sentence_range(text_list, sentence_range=(1, 4)):
    """Chunk list of sentences into blocks of a specified number of concatenated sentences."""
    it = iter(text_list)
    while True:
        chunk = tuple(itertools.islice(it, random.randint(sentence_range[0], sentence_range[1])))
        if not chunk:
            break
        yield " ".join(chunk)


def chunk_text_as_list_by_sentence_range(text_list, sentence_range=(2, 4)):
    """Chunk list of sentences into blocks of 2 to 4 sentences (as a list)."""
    it = iter(text_list)
    while True:
        chunk = list(itertools.islice(it, random.randint(sentence_range[0], sentence_range[1])))
        if not chunk:
            break
        yield chunk


def chunk_text_n_block_as_str(text_list, n):
    """Chunk list of sentences into n roughly equal blocks of concatenated sentences."""
    if n <= 1:
        return [" ".join(text_list)]
    total_sentences = len(text_list)
    sentences_per_block = max(1, (total_sentences + n - 1) // n)
    return [" ".join(text_list[i:i + sentences_per_block]) 
            for i in range(0, total_sentences, sentences_per_block)]

def chunk_text_n_block_as_list(text_list, n):
    """Chunk list of sentences into n roughly equal blocks of sentences (as a list)."""
    if n <= 1:
        return [text_list]
    total_sentences = len(text_list)
    sentences_per_block = max(1, (total_sentences + n - 1) // n)
    return [text_list[i:i + sentences_per_block] 
            for i in range(0, total_sentences, sentences_per_block)]


def generate_instruction_data(
    example: dict, 
    task_type: str, 
    batched: bool = False, 
    loose_chunking: bool = False,
    loose_chunking_token_count: int = 512,
    use_only_generation: bool = False,
    **kwargs
):
    global _stored_sample
    prev_example = _stored_sample
    _stored_sample = None  # reset

    # Preprocess the example if batched: split first text into sentences.
    if batched:
        sentences = bf.text_to_sentences(example["text"][0]).split("\n")
        example["text"] = [s for s in sentences if s.strip()]

    # Store current example for later use.
    _stored_sample = example.copy()

    # Determine task type if not specified.
    if task_type is None:
        task_type = get_default_task_type(
            example, 
            batched, 
            prev_example, 
            use_only_generation,
        )

    # Apply chunking based on task type.
    if batched:
        if loose_chunking:
            if task_type in ("half_completion", "char_identification", "random", 
                             "typo_correction", "mlm", "shuffle", "space_insertion",
                             "one_word_completion", "char_count", "token_type"):
                n = (example["metadata"][0]["token_count"] // loose_chunking_token_count) + 1
                example["text"] = chunk_text_n_block_as_str(example["text"], n)
            elif task_type == "passage_completion":
                n = (example["metadata"][0]["token_count"] // loose_chunking_token_count * 2) + 1
                temp_text = []
                for chunk in chunk_text_n_block_as_str(example["text"], n):
                    words = chunk.split(" ")
                    n = len(words) // 3
                    temp_text.append([
                        " ".join(words[:n]),
                        " ".join(words[n:2*n]),
                        " ".join(words[2*n:]),
                    ])
                example["text"] = temp_text
            elif task_type in ("last_phrase_prediction", "sentence_reordering", "sentence_deletion"):
                n = (example["metadata"][0]["token_count"] // loose_chunking_token_count) + 1
                example["text"] = chunk_text_n_block_as_list(example["text"], n)
            else:
                raise ValueError(f"Unsupported task type for loose chunking: {task_type}")
        else:
            if task_type in ("half_completion", "char_identification", "random", 
                             "typo_correction", "mlm", "shuffle", "space_insertion",
                             "one_word_completion", "char_count", "token_type"):
                example["text"] = list(chunk_text_as_str_by_sentence_range(example["text"], kwargs.get("sentence_range", (5, 10))))
            elif task_type == "passage_completion": # Use the same chunking as in loose_chunking (to simulate the Fill-in-the-Middle task used in DeepSeek V3).
                n = (example["metadata"][0]["token_count"] // loose_chunking_token_count * 2) + 1
                temp_text = []
                for chunk in chunk_text_n_block_as_str(example["text"], n):
                    words = chunk.split(" ")
                    n = len(words) // 3
                    temp_text.append([
                        " ".join(words[:n]),
                        " ".join(words[n:2*n]),
                        " ".join(words[2*n:]),
                    ])
                example["text"] = temp_text
            elif task_type in ("last_phrase_prediction", "sentence_reordering", "sentence_deletion"):
                example["text"] = list(chunk_text_as_list_by_sentence_range(example["text"], kwargs.get("sentence_range", (5, 10))))
    
    # if prev_example is None, set it to the current example.
    if prev_example is None:
        prev_example = _stored_sample

    # Map task types to their functions.
    task_function_mapping = {
        "shuffle": lambda ex: generate_shuffle_instruction_data(ex, batched, **kwargs),
        "mlm": lambda ex: generate_mlm_instruction_data(ex, batched, **kwargs),
        "char_identification": lambda ex: generate_char_identification_instruction_data(ex, batched, **kwargs),
        "random": lambda ex: generate_random_word_instruction_data(ex, batched, **kwargs),
        "token_type": lambda ex: generate_token_type_instruction_data(ex, batched, **kwargs),
        "half_completion": lambda ex: generate_half_completion_instruction_data(ex, batched, **kwargs),
        "typo_correction": lambda ex: generate_typo_correction_instruction_data(ex, batched, **kwargs),
        "sentence_reordering": lambda ex: generate_sentence_reordering_instruction_data(ex, batched, **kwargs),
        "sentence_deletion": lambda ex: generate_sentence_deletion_instruction_data(ex, batched, prev_example, **kwargs),
        "last_phrase_prediction": lambda ex: generate_last_phrase_prediction_instruction_data(ex, batched, prev_example, **kwargs),
        "passage_completion": lambda ex: generate_passage_completion_instruction_data(ex, batched, **kwargs),
        "one_word_completion": lambda ex: generate_one_word_completion_instruction_data(ex, batched, **kwargs),
        "char_count": lambda ex: generate_char_count_instruction_data(ex, batched, **kwargs),
        "space_insertion": lambda ex: generate_space_insertion_instruction_data(ex, batched, **kwargs),
    }

    # Execute the task function based on the task type.
    if task_type in task_function_mapping:
        return task_function_mapping[task_type](example)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
