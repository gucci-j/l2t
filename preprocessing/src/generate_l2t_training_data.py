from datasets import Dataset, load_dataset, concatenate_datasets
from functools import partial
from itertools import chain
from pathlib import Path
from queue import PriorityQueue
import random
import numpy as np
import copy
from transformers import AutoTokenizer

from tasks import generate_instruction_data
from util import save_to_disk

def format_dataset(
    train_dataset: Dataset,
    dev_dataset: Dataset,
    num_workers: int = 4,
    no_anchor_words: bool = False,
):
    anchor_words = [
        "Answer:", "Response:", "A:", "(A)", "A)", "A.", ""
    ]
    num_printed = 0
    def _format_sample(sample: dict) -> dict:
        nonlocal num_printed
        
        if not no_anchor_words:
            # Concatenate the prompt and completion texts
            anchor_word = random.choice(anchor_words)
        else:
            anchor_word = ""
        if not sample["completion"].startswith(" ") and anchor_word != "":
            sample["text"] = sample["prompt"] + "\n\n" + anchor_word + " " + sample["completion"]
        else:
            sample["text"] = sample["prompt"] + "\n\n" + anchor_word + sample["completion"] 

        # sanity check
        if random.random() < 0.01 and num_printed < 100:
            print('\n'+ '-'*100  +f'\n{sample["text"]}')
            num_printed += 1
        
        return sample

    train_dataset = train_dataset.map(
        _format_sample, 
        remove_columns=train_dataset.column_names,
        num_proc=num_workers
    )
    if dev_dataset is not None:
        dev_dataset = dev_dataset.map(
            _format_sample, 
            remove_columns=dev_dataset.column_names,
            num_proc=num_workers
        )
    
    return train_dataset, dev_dataset


def spfhp(seq_lens, chunk_length=2048):
    q = PriorityQueue()
    q.put((0,[]))
    idx = seq_lens.argsort()[::-1]
    for i in idx:
        n, pack = seq_lens[i], q.get()
        if n+pack[0] > chunk_length:
            q.put(pack)
            pack = (0,[])
        q.put((n+pack[0], pack[1]+[i]))
    return list(q.queue)


def pack(
    sample, 
    chunk_length=2048, 
    pad_token_id=0
):
    # Convert the sample into a dictionary
    sample = {k: sample[k] for k in sample.keys()}

    # compute packing arrangement
    seq_lens = np.array([len(t) for t in sample["input_ids"]])
    chunks = spfhp(seq_lens, chunk_length=chunk_length)
    random.shuffle(chunks)

    # pack sequences according to arrangement
    result = {}
    for k in sample.keys():
        pad_id = pad_token_id if k == "input_ids" else 0
        result[k] = [
            list(chain(*[sample[k][i] for i in chunk[1]], [pad_id] * (chunk_length - chunk[0]))
            ) for chunk in chunks
        ]

    # add labels (same as input_ids!)
    result["labels"] = result["input_ids"].copy()
    return result


def generate_samples_on_the_fly(
    dataset: Dataset,
    num_workers: int = 4,
    batched: bool = False,
    task_type: str = None,
    min_num_words: int = 5,
    use_only_primary: bool = True,
    sentence_range: tuple = (5, 8),
    no_instruction_variation: bool = False,
    loose_chunking: bool = False,
    loose_chunking_token_count: int = 512,
    use_only_generation: bool = True,
    no_instruction: bool = True,
    include_starting_word: bool = True,
) -> Dataset:
    # Generate the instruction data
    generate_instruction_data_fn = partial(
        generate_instruction_data, 
        task_type=task_type, 
        batched=batched, 
        loose_chunking=loose_chunking,
        loose_chunking_token_count=loose_chunking_token_count,
        min_num_words=min_num_words, 
        use_only_primary=use_only_primary,
        sentence_range=sentence_range,
        no_instruction_variation=no_instruction_variation,
        use_only_generation=use_only_generation,
        no_instruction=no_instruction,
        include_starting_word=include_starting_word,
    )
    return dataset.map(
        generate_instruction_data_fn,
        num_proc=num_workers,
        remove_columns=dataset.column_names,
        batched=batched,
        batch_size=1
    )


def group_texts(examples: dict, block_size=128):
    # Concatenate all texts.
    try:
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    except Exception:
        print(examples)
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main(args):
    # Load the dataset
    print("Loading dataset...")
    # Load the dataset
    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus", 
        "fineweb-edu-dedup", 
        split="train", 
        num_proc=args.num_workers
    ) # 220B tokens
    dataset = dataset.shard(num_shards=40, index=args.shard_index) # 5.5B tokens
    
    # Split the dataset into train and test
    if args.shard_index == 0:
        dataset = dataset.train_test_split(test_size=0.1, shuffle=True)
        train_dataset = dataset["train"]
        dev_dataset = dataset["test"]
        del dataset
    else:
        train_dataset = dataset
        dev_dataset = None

    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Mix the NTP samples with the training samples
    if args.mix_ntp_samples:
        ntp_tokenizer = AutoTokenizer.from_pretrained(
            args.ntp_tokenizer_name_or_path,
            cache_dir=args.cache_dir
        )
        if ntp_tokenizer.pad_token is None:
            ntp_tokenizer.pad_token = ntp_tokenizer.eos_token

        if args.duplicate_ntp_samples:
            # Duplicate samples
            train_ntp_dataset = copy.deepcopy(train_dataset)
            if dev_dataset is not None:
                dev_ntp_dataset = copy.deepcopy(dev_dataset)
        else:
            # Split the dataset into L2T and NTP datasets
            train_dataset = train_dataset.train_test_split(
                test_size=args.mix_ntp_ratio, shuffle=True
            )
            train_dataset, train_ntp_dataset = train_dataset["train"], train_dataset["test"]
            if dev_dataset is not None:
                dev_dataset = dev_dataset.train_test_split(
                    test_size=args.mix_ntp_ratio, shuffle=True
                )
                dev_dataset, dev_ntp_dataset = dev_dataset["train"], dev_dataset["test"]
                
        # Tokenize the dataset
        print("Tokenizing the NTP dataset...")
        train_ntp_dataset = train_ntp_dataset.map(
            lambda examples: ntp_tokenizer(examples["text"]),
            batched=True,
            num_proc=args.num_workers,
            remove_columns=train_ntp_dataset.column_names,
        )
        if dev_dataset is not None:
            dev_ntp_dataset = dev_ntp_dataset.map(
                lambda examples: ntp_tokenizer(examples["text"]),
                batched=True,
                num_proc=args.num_workers,
                remove_columns=dev_ntp_dataset.column_names,
            )
            
        # Group the texts
        print("Grouping the NTP texts...")
        train_ntp_dataset = train_ntp_dataset.map(
            lambda examples: group_texts(examples, args.max_length),
            batched=True,
            num_proc=args.num_workers,
        )
        if dev_dataset is not None:
            dev_ntp_dataset = dev_ntp_dataset.map(
                lambda examples: group_texts(examples, args.max_length),
                batched=True,
                num_proc=args.num_workers,
            )
            
        # Filter out examples with length not equal to max_length
        print("Filtering out examples with length not equal to max_length...")
        train_ntp_dataset = train_ntp_dataset.filter(
            lambda example: len(example["input_ids"]) == args.max_length, num_proc=args.num_workers
        )
        if dev_dataset is not None:
            dev_ntp_dataset = dev_ntp_dataset.filter(
                lambda example: len(example["input_ids"]) == args.max_length, num_proc=args.num_workers
            )
        
    # Generate samples on the fly if needed
    print("Generating samples on the fly...")
    if args.generate_samples_on_the_fly:
        # Define common parameters for both datasets
        generation_params = {
            "num_workers": args.num_workers,
            "batched": args.batched,
            "task_type": args.task_type,
            "loose_chunking": args.loose_chunking,
            "loose_chunking_token_count": args.loose_chunking_token_count,
            "min_num_words": args.min_num_words,
            "use_only_primary": True,
            "sentence_range": args.sentence_range,
            "no_instruction_variation": args.no_instruction_variation,
            "use_only_generation": True,
            "no_instruction": True,
            "include_starting_word": True,
        }
        # Process train dataset
        train_dataset = generate_samples_on_the_fly(train_dataset, **generation_params)
        # Process dev dataset if available
        if dev_dataset is not None:
            dev_dataset = generate_samples_on_the_fly(dev_dataset, **generation_params)

    # Format the dataset
    print("Formatting the dataset...")
    train_dataset, dev_dataset = format_dataset(
        train_dataset, 
        dev_dataset, 
        args.num_workers // 2,
        args.no_anchor_words
    )

    # Tokenize the dataset
    print("Tokenizing the dataset...")
    train_dataset = train_dataset.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        num_proc=args.num_workers,
        remove_columns=train_dataset.column_names,
    )
    if dev_dataset is not None:
        dev_dataset = dev_dataset.map(
            lambda examples: tokenizer(examples["text"]),
            batched=True,
            num_proc=args.num_workers,
            remove_columns=dev_dataset.column_names,
        )
    
    # Filter out rows of tokenized_dataset that are too long
    print("Filtering out rows of tokenized_dataset that are too long...")
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids"]) <= args.max_length, 
        num_proc=args.num_workers
    )
    if dev_dataset is not None:
        dev_dataset = dev_dataset.filter(
            lambda x: len(x["input_ids"]) <= args.max_length,
            num_proc=args.num_workers
        )
    
    # Group the texts
    print("Grouping the texts...")
    pack_fn = partial(pack, chunk_length=args.max_length, pad_token_id=tokenizer.pad_token_id)
    train_dataset = train_dataset.map(pack_fn, batched=True, num_proc=args.num_workers)
    if dev_dataset is not None:
        dev_dataset = dev_dataset.map(pack_fn, batched=True, num_proc=args.num_workers)
        
    # Concatenate NTP samples with the training samples
    if args.mix_ntp_samples:
        train_dataset = concatenate_datasets([train_dataset, train_ntp_dataset])
        if dev_dataset is not None:
            dev_dataset = concatenate_datasets([dev_dataset, dev_ntp_dataset])

    # Shuffle the dataset
    print("Shuffling the dataset...")
    train_dataset = train_dataset.shuffle(seed=21)
    if dev_dataset is not None:
        dev_dataset = dev_dataset.shuffle(seed=21)

    # Save the tokenized dataset to a file
    print("Saving tokenized dataset...")
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir()
    save_to_disk(train_dataset, Path(args.output_dir) / str(args.shard_index) / "train", num_proc=args.num_workers)
    if dev_dataset is not None:
        save_to_disk(dev_dataset, Path(args.output_dir) / str(args.shard_index) / "dev", num_proc=args.num_workers)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Path to the output data directory"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default=None,
        help="Path to the cache directory"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        required=True,
        help="Name or path of the tokenizer to use"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4, 
        help="Number of worker processes to use"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512, 
        help="Maximum length of the tokenized sequences"
    )
    parser.add_argument(
        "--mix_ntp_samples",
        action="store_true",
        help="Whether to mix NTP samples with the training samples"
    )
    parser.add_argument(
        "--mix_ntp_ratio",
        type=float,
        default=0.1,
        help="[mix_ntp_samples] Ratio of NTP samples to mix with the training samples."
    )
    parser.add_argument(
        "--duplicate_ntp_samples",
        action="store_true",
        help="Whether to duplicate NTP samples."
    )
    parser.add_argument(
        "--ntp_tokenizer_name_or_path",
        type=str,
        default=None,
        help="[mix_ntp_samples] Name or path of the NTP tokenizer to use"
    )
    parser.add_argument(
        "--no_anchor_words",
        action="store_true",
        help="Whether to use anchor words in the prompt. If True, the prompt will be used as is.",
    )
    
    # generate_samples_on_the_fly
    parser.add_argument(
        "--generate_samples_on_the_fly",
        action="store_true",
        help="Whether to generate samples on the fly",
    )
    parser.add_argument(
        "--batched",
        action="store_true",
        help="[generate_samples_on_the_fly] Whether to generate samples in batches. This allows us to generate samples with varying lengths.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        help="[generate_samples_on_the_fly] The type of task to generate samples for. If None, all tasks are considered.",
    )
    parser.add_argument(
        "--min_num_words",
        type=int,
        default=5,
        help="[generate_samples_on_the_fly] Minimum number of words in the text for generating samples.",
    )
    parser.add_argument(
        "--sentence_range",
        type=int,
        nargs=2,
        default=(5, 10),
        help="[generate_samples_on_the_fly] Range of the number of sentences to concatenate for generating samples.",
    )
    parser.add_argument(
        "--no_instruction_variation",
        action="store_true",
        help="[generate_samples_on_the_fly] Whether to generate samples without instruction variations.",
    )
    parser.add_argument(
        "--loose_chunking",
        action="store_true",
        help="[generate_samples_on_the_fly] Whether to use loose chunking.",
    )
    parser.add_argument(
        "--loose_chunking_token_count",
        type=int,
        default=512,
        help="[generate_samples_on_the_fly] Token count for loose chunking.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Index of the shard to use",
    )
    
    args = parser.parse_args()
    main(args)
