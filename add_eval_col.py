from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from bert_score import BERTScorer
import nltk
#nltk.download('wordnet')
#nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import pandas as pd 
import statistics

# BERTScore: Evaluates semantic similarity using embeddings.
# BERTScore Cosine Similarity: Measures the cosine similarity of embeddings.
# BLEU Score: Focuses on n-gram precision with brevity penalty.
# ROUGE-1, ROUGE-2, ROUGE-L: Measures n-gram and sequence overlaps.
# METEOR Score: Combines precision, recall, stemming, synonyms, and word order.
# All except cosine similarity range from 0 to 1, with scores closer to 1 indicating higher similarity


# other aggregation methods - variance? max, min? 



def calculate_bert_cosine_similarity(reference_tokens, candidate, tokenizer, model): 
    candidate_tokens = tokenizer(candidate, return_tensors='pt')
    
    with torch.no_grad():
        reference_embeddings = model(**reference_tokens).last_hidden_state
        candidate_embeddings = model(**candidate_tokens).last_hidden_state
        
    # Compute cosine similarity between the embeddings
    similarity = torch.nn.functional.cosine_similarity(
        candidate_embeddings.mean(dim=1),
        reference_embeddings.mean(dim=1)
    )
        
    return similarity.item()

def calculate_meteor_score(reference_tokens, candidate): 
    candidate_tokens = nltk.word_tokenize(candidate)
    score = meteor_score([reference_tokens], candidate_tokens)
    return score 


def add_eval_col(reference_list, filename, download_path, new_name): 

    df = pd.read_csv(filename)

    all_candidates = df['Utterance'].tolist()

    print('starting')


    #cosine similarity  
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #bert_reference_tokens = bert_tokenizer(reference, return_tensors="pt", padding=True, truncation=True)
    bert_reference_token_list = [bert_tokenizer(reference, return_tensors="pt", padding=True, truncation=True) for reference in reference_list]
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    print('cosine similarity reference tokenizing done')
    
    cosine_similarities = []
    for candidate in all_candidates: 
        cosine_similarity_list = []
        for bert_reference_token in bert_reference_token_list: 
            cosine_similarity_list.append(calculate_bert_cosine_similarity(bert_reference_token, candidate, bert_tokenizer, bert_model))
        cosine_similarities.append(statistics.mean(cosine_similarity_list))

    #cosine_similarities = [calculate_bert_cosine_similarity(bert_reference_tokens, candidate, bert_tokenizer, bert_model) for candidate in all_candidates]
    df['Cosine Similarity'] = cosine_similarities
    print('cosine similarity done\n')




    #bertscore f1 
    bertscorer = BERTScorer(model_type='bert-base-uncased')

    bertscore_f1 = []
    for candidate in all_candidates: 
        P, R, F1 = bertscorer.score([candidate] * len(reference_list), reference_list)
        avg_F1 = F1.mean().item()
        bertscore_f1.append(avg_F1)

    #P, R, F1 = bertscorer.score(all_candidates, [reference] * len(all_candidates))
    #bertscore_f1 = F1.tolist()

    df['BERTScore F1'] = bertscore_f1
    print('bertscore f1 done\n')




    #rouge 
    rougescorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    #rougeL_f1 = [rougescorer.score(reference, candidate)['rougeL'].fmeasure for candidate in all_candidates]

    rougeL_f1 = []
    for candidate in all_candidates: 
        rougescores = []
        for reference in reference_list: 
            rougescore = rougescorer.score(reference, candidate)['rougeL'].fmeasure 
            rougescores.append(rougescore)
        rougeL_f1.append(statistics.mean(rougescores))

    df['rougeL F1'] = rougeL_f1
    print('rouge f1 done\n')




    #meteor 
    #meteor_reference_tokens = nltk.word_tokenize(reference)
    #meteor_scores = [calculate_meteor_score(meteor_reference_tokens, candidate) for candidate in all_candidates]
    
    meteor_reference_token_list = [nltk.word_tokenize(reference) for reference in reference_list]
    meteor_scores = []
    for candidate in all_candidates: 
        meteorlist = []
        for reference_token in meteor_reference_token_list: 
            meteorscore = calculate_meteor_score(reference_token, candidate)
            meteorlist.append(meteorscore)
        meteor_scores.append(statistics.mean(meteorlist))

    df['METEOR Score'] = meteor_scores
    print('meteor done\n')

    df.to_csv(download_path + new_name + '.csv')

f = 'ZMU_New_NY_4November2022_interest_driven'
filename = 'interview_csvs/' + f + '.csv'

download_path = '/Users/haoningjiang/Desktop/exploratory-research/evalcol_dest/'
new_name = 'ZMU_New_NY_4November2022_EvalCol_multiref'

#reference = "But then you meet a group of people that you have a ton in common with, and you have similar goals, and it's like, yeah, these guys are really cool"

reference_list = [
    "But usually what I would do, and a couple of the other kids I, um, were friendly with, were, we'd usually just return back to the building and then work on homework or start other projects", 
    "But then you meet a group of people that you have a ton in common with, and you have similar goals, and it's like, yeah, these guys are really cool", 
    "So being able to get into the program and then surround yourself with, like, other people from a similar background to you that are also interested in art, which wasn't really common, was really, it was fun", 
    "Um, okay, well, I got to meet friends that I, I consider like lifetime, lifelong friends at this point",
    "Um, some of them we just play video games with together, and, like, that's it.",
    "So being able to get into the program and then surround yourself with, like, other people from a similar background to you that are also interested in art, which wasn't really common, was really, it was fun",
    ]


add_eval_col(reference_list, filename, download_path, new_name)

