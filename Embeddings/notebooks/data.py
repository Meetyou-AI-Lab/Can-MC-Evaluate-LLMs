from config import *
from conversation import get_prompt
from utils import *

import pandas as pd
from datasets import load_from_disk
from os.path import join, exists
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Union

def load_iti_nq_open_val_as_df(if_save: bool=False) -> pd.DataFrame:
    """ Return a dataframe with three columns: <question, answer, label> """
    dataset_path = join(DATA, 'iti_nq_open_val')
    dataset = load_from_disk(dataset_path)
    questions = dataset['question']
    answers = dataset['answer']
    false_answers = dataset['false_answer']
    assert len(answers) == len(questions)
    all_questions = []
    all_answers = []
    all_labels = []
    for question, answer, false_answer in zip(questions, answers, false_answers):
        for ans in answer:
            all_questions.append(question.strip())
            all_answers.append(ans.strip())
            all_labels.append(True) 
        all_questions.append(question.strip())
        all_answers.append(false_answer.strip())
        all_labels.append(False)
    dataset = pd.DataFrame({
        'question': all_questions,
        'answer'  : all_answers,
        'label'   : all_labels})
    if not exists(join(dataset_path, 'TruthfulQA_original.tsv')) and if_save:
        dataset.to_csv(join(dataset_path, 'TruthfulQA_original.tsv'), sep='\t', index=False)
    return dataset

def load_tqa_as_df(
        version: str='v0',
        if_with_best_answer: bool=False,
        if_save: bool=False) -> pd.DataFrame:
    """ Load the original TruthfulQA dataset. Format:
        <Type, Category, Question, Best Answer, Correct Answers, Incorrect Answers, Source>

    Args:
        version (str, optional): Version number, either be 'v0' or 'v1'. Defaults to 'v0'.
        if_with_best_answer (bool, optional): Whether include the best answers as the true
            answers. Defaults to False.
        if_save (bool, optional): Whether save a copy of the output 'TruthfulQA_original.tsv' to local or not
            for efficiency. Defaults to False.

    Returns:
        pd.DataFrame: The collection of TruthfulQA QA pairs; each sample has the format of
            <question, answer, label>.
    """
    assert version in ['v1', 'v0'], f'Version {version} is not valid.'
    file_name = f'TruthfulQA.csv'
    dataset_path = join(DATA, 'TruthfulQA', version)
    src_path = join(dataset_path, file_name)
    dataset = pd.read_csv(src_path, sep=',', header=0)
    types = dataset['Type'].tolist()
    categories = dataset['Category'].tolist()
    questions = dataset['Question'].tolist()
    best_answers = dataset['Best Answer'].tolist()
    correct_answers = dataset['Correct Answers'].tolist()
    false_answers = dataset['Incorrect Answers'].tolist()
    correct_answers = [ans.split(';') for ans in correct_answers]
    false_answers = [ans.split(';') for ans in false_answers]
    sources = dataset['Source'].tolist()
    assert len(correct_answers) == len(false_answers)
    all_types = []
    all_categories = []
    all_questions = []
    all_answers = []
    all_labels = []
    all_sources = []
    for (
            type_, category, question,
            best_answer, answer,
            false_answer, source
        ) in zip(
            types, categories, questions,
            best_answers, correct_answers,
            false_answers, sources
        ):
        true_answer = [best_answer] + answer if if_with_best_answer else answer
        for ans in true_answer:
            all_types.append(type_)
            all_categories.append(category)
            all_questions.append(question.strip())
            all_answers.append(ans.strip())
            all_labels.append(True)
            all_sources.append(source)
        for ans in false_answer:
            all_types.append(type_)
            all_categories.append(category)
            all_questions.append(question.strip())
            all_answers.append(ans.strip())
            all_labels.append(False)
            all_sources.append(source)
    dataset = pd.DataFrame({
        'type'    : all_types,
        'category': all_categories,
        'question': all_questions,
        'answer'  : all_answers,
        'true'    : all_labels,
        'source'  : all_sources})
    if not exists(join(dataset_path, 'TruthfulQA_original.tsv')) and if_save:
        dataset.to_csv(join(dataset_path, 'TruthfulQA_original.tsv'), sep='\t', index=False)
    return dataset

def load_tqa_ft_as_df(
        version: str='v0',
        if_save: bool=False) -> pd.DataFrame:
    dataset_path = join(DATA, 'TruthfulQA', version)
    info_jsonl_path = join(dataset_path, 'finetune_info.jsonl')
    truth_jsonl_path = join(dataset_path, 'finetune_truth.jsonl')
    info_jsonl = load_jsonl(info_jsonl_path)
    truth_jsonl = load_jsonl(truth_jsonl_path)
    info_df = pd.DataFrame.from_dict(jsonl_to_dict(info_jsonl))
    truth_df = pd.DataFrame.from_dict(jsonl_to_dict(truth_jsonl))
    info_df["prompt"] = info_df['prompt'].apply(lambda x: x.replace('\nHelpful:', ''))
    truth_df["prompt"] = truth_df['prompt'].apply(lambda x: x.replace('\nTrue:', ''))
    info_prompt_to_completion = columns_to_dict(info_df, 'prompt', 'completion')
    truth_prompt_to_completion = columns_to_dict(truth_df, 'prompt', 'completion')
    info_truth_dict = {'prompt': [], 'helpful': [], 'true': []}
    for i in info_prompt_to_completion:
        info_truth_dict['prompt'].append(i)
        info_truth_dict['helpful'].append(info_prompt_to_completion[i])
        info_truth_dict['true'].append(truth_prompt_to_completion[i])
    info_truth_df = pd.DataFrame.from_dict(info_truth_dict)
    info_truth_df['question'] = info_truth_df['prompt'].apply(lambda x: x.split('\nA: ')[0][3:])
    info_truth_df['answer'] = info_truth_df['prompt'].apply(lambda x: x.split('\nA: ')[1])
    info_truth_df = info_truth_df.drop(['prompt'], axis=1)
    for col in info_truth_df:
        info_truth_df[col] = info_truth_df[col].apply(text_normalize_)
    info_truth_df['helpful'] = info_truth_df['helpful'].apply(lambda x: x == 'yes')
    info_truth_df['true'] = info_truth_df['true'].apply(lambda x: x == 'yes')
    if not exists(join(dataset_path, 'TruthfulQA_finetune.tsv')) and if_save:
        info_truth_df.to_csv(join(dataset_path, 'TruthfulQA_finetune.tsv'), sep='\t', index=False)
    return info_truth_df

def load_dataset_fast(dataset_name: str='tqa', **kwargs) -> pd.DataFrame:
    version = kwargs.get('version', 'v0')
    data_type = kwargs.get('data_type', 'finetune')
    assert data_type in ['finetune', 'original']
    if dataset_name == 'tqa':
        src_name = 'TruthfulQA'
        dataset_path = join(DATA, src_name, version)
    else:
        raise NotImplementedError
    file_path = join(dataset_path, f'TruthfulQA_{data_type}.tsv')
    if not exists(file_path):
        if dataset_name == 'tqa':
            if data_type == 'original':
                load_tqa_as_df(version=version, if_save=True)
            else:
                load_tqa_ft_as_df(version=version, if_save=True)
        else:
            raise NotImplementedError
    dataset = pd.read_csv(file_path, sep='\t', header=0)
    return dataset

def load_curated_tqa(seed=42, **kwargs) -> pd.DataFrame:
    version = kwargs.get('version', 'v0')
    data_type = kwargs.get('data_type', 'finetune')
    assert data_type in ['finetune', 'original']
    data = load_dataset_fast(version=version, data_type=data_type)
    TT = data[(data['true'] == True) & (data['helpful'] == True)]
    TF = data[(data['true'] == True) & (data['helpful'] == False)]
    FT = data[(data['true'] == False) & (data['helpful'] == True)]
    FF = data[(data['true'] == False) & (data['helpful'] == False)]
    size = min(len(TT), len(TF), len(FT), len(FF))
    TT_curated = TT.loc[random_choose(TT.index, size=size, seed=seed)]
    TF_curated = TF.loc[random_choose(TF.index, size=size, seed=seed)]
    FT_curated = FT.loc[random_choose(FT.index, size=size, seed=seed)]
    FF_curated = FF.loc[random_choose(FF.index, size=size, seed=seed)]
    curated_data = pd.concat([TT_curated, TF_curated, FT_curated, FF_curated], axis=0)
    curated_data['category'] = ['TT'] * size + ['TF'] * size + ['FT'] * size + ['FF'] * size
    curated_data = curated_data.sample(frac=1, random_state=seed)
    curated_data = curated_data[['category']]
    return curated_data

def load_curated_tqa_val_indices(
        curated_data: pd.DataFrame,
        val_size: float,
        seed: int=42,
        shuffle: bool=True) -> np.ndarray:
    val_indices = []
    TT_indices = curated_data[curated_data['category'] == 'TT'].index
    TF_indices = curated_data[curated_data['category'] == 'TF'].index
    FT_indices = curated_data[curated_data['category'] == 'FT'].index
    FF_indices = curated_data[curated_data['category'] == 'FF'].index
    for i in [TT_indices, TF_indices, FT_indices, FF_indices]:
        val_indices.append(random_choose(i, int(len(i) * val_size), seed))
    val_indices = np.concatenate(val_indices, axis=0)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(val_indices)
    return val_indices

class LlmDataset(Dataset):
    def __init__(self, df: pd.DataFrame, model_name: str, tokenizer: Union[None, PreTrainedTokenizer]= None):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        if tokenizer is None:
            self.tokenizer = load_tokenizer(model_name)
        self.data: pd.DataFrame = df
        self.model_name: str = model_name
        self.questions: List[str] = self.data['question'].tolist()
        self.answers: List[str] = self.data['answer'].tolist()
        self.labels: List[bool] = self.data['label'].tolist()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        question = self.questions[index]
        answer = self.answers[index]
        label = self.labels[index]
        processed_question = get_prompt(question, MODEL_MAPPING[self.model_name])['prompt']
        processed_question_ids = self.tokenizer(processed_question, return_tensors='pt').input_ids
        prompt = f'{processed_question} {answer}'
        prompt_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
        return {
            'prompt_ids'            : prompt_ids,
            'prompt'                : prompt,
            'label'                 : label,
            'processed_question'    : processed_question, # processed q
            'processed_question_ids': processed_question_ids,
            'original_question'     : question,
            'original_answer'       : answer}