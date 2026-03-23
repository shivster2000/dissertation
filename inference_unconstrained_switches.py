import stanza
import argparse
import json
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Mistral3ForConditionalGeneration, BitsAndBytesConfig
import bitsandbytes as bnb
from datetime import datetime

from utils import read_file, read_pickle

def init():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', type=str, required=True, choices=['baseline', 'matrix', 'equivalence'])
    arg_parser.add_argument('--lang1', type=str, default='en')
    arg_parser.add_argument('--lang2', type=str, default='hi')
    arg_parser.add_argument('--matrix_lang', type=str, default='hi', 
                           help='Matrix language (provides grammatical frame)')
    arg_parser.add_argument('--src', type=str, default='')
    arg_parser.add_argument('--tgt', type=str, default='')
    arg_parser.add_argument('--src_translated', type=str, default='')
    arg_parser.add_argument('--tgt_translated', type=str, default='')
    arg_parser.add_argument('--gold_align', type=str, default='')
    arg_parser.add_argument('--model_id', type=str)
    arg_parser.add_argument('--output', type=str)
    arg_parser.add_argument('--download_stanza', action='store_true',
                           help='Download Stanza models on first run')
    arg_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    eot_token = {
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "<|eot_id|>",
        "mistralai/Ministral-3-8B-Instruct-2512-BF16": "</s>",
        "google/gemma-2-9b-it": "<end_of_turn>"
    }

    args = arg_parser.parse_args()
    args.eot_token = eot_token[args.model_id]
    
    return args

def log_to_file(filename, data):
    """Appends a dictionary as a JSON line to the specified file."""
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

class MLFAnalyser:
    def __init__(self, lang1='hi', lang2='en', download_models=True):
        self.lang1 = lang1
        self.lang2 = lang2
        if download_models:
            stanza.download(lang1)
            stanza.download(lang2)

        self.debug = args.debug

        self.nlp_models = {
            lang1: stanza.Pipeline(lang1, processors='tokenize,pos,lemma,depparse', verbose=False),
            lang2: stanza.Pipeline(lang2, processors='tokenize,pos,lemma,depparse', verbose=False)
        }

        # Define content morpheme POS tags (can be from EL)
        self.content_pos_tags = ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV', 'NUM']

        # Define system morpheme dependency relations with EXTERNAL relations
        # These MUST come from Matrix Language
        self.external_dep_relations = {
            # Grammatical relations that cross constituent boundaries
            'case',      # Case markers (relate NP to verb)
            'aux',       # Auxiliaries (agree with subject, scope over VP)
            'cop',       # Copulas (link subject and predicate)
            'mark',      # Complementizers/subordinators (link clauses)
            'cc',        # Coordinating conjunctions (link constituents)
            'det',       # Determiners (some theories treat as external)
            'clf',       # Classifiers
            'discourse', # Discourse markers
        }

        self.external_pos_tags = {
            'ADP',   # Adpositions (prepositions/postpositions)
            'AUX',   # Auxiliaries
            'SCONJ', # Subordinating conjunctions
            'CCONJ', # Coordinating conjunctions
            'PART',  # Particles (often grammatical)
        }

    def tag_morphemes(self, sentence, lang):
        nlp = self.nlp_models[lang]
        doc = nlp(sentence)
        morpheme_tags = []
        for sent in doc.sentences:
            for word in sent.words:
                morpheme_tags.append({
                    'text': word.text,
                    'lemma': word.lemma,
                    'index': word.id - 1,
                    'upos': word.upos,
                    'xpos': word.xpos,
                    'feats': word.feats,
                    'head': word.head - 1 if word.head > 0 else -1,
                    'deprel': word.deprel,
                    'type': 'content' if word.upos in self.content_pos_tags else 'system',
                    'has_external_relations': self.has_external_relations(word, lang),
                    'lang': lang
                })
        return morpheme_tags

    def has_external_relations(self, word, lang):
        # Check by dependency relations
        if word.deprel in self.external_dep_relations:
            return True
        # Check by POS tag
        if word.upos in self.external_pos_tags:
            return True

    """ No constituent boundary checking yet """

    def get_valid_mlf_switches(self, ml_sent, el_sent, alignment, ml_lang, el_lang):
        ml_morphemes = self.tag_morphemes(ml_sent, ml_lang)
        el_morphemes = self.tag_morphemes(el_sent, el_lang)

        align_pairs = [tuple(map(int, pair.split('-'))) for pair in alignment.strip().split()]

        valid_el_words = []
        ml_frame_words = []
        
        # NEW: Sets to keep track of indices we have already successfully added
        seen_ml_indices = set()
        seen_el_indices = set()

        for ml_idx, el_idx in align_pairs:
            if ml_idx >= len(ml_morphemes) or el_idx >= len(el_morphemes):
                continue
        
            ml_morph = ml_morphemes[ml_idx]
            el_morph = el_morphemes[el_idx]

            """ The System Morpheme Principle """
            # Only check this ML word if we haven't already added it to the frame list
            if ml_idx not in seen_ml_indices:
                if ml_morph['type'] == 'system' and ml_morph['has_external_relations']:
                    ml_frame_words.append({
                        'word': ml_morph['text'],
                        'reason': f"External system morpheme ({ml_morph.get('deprel', ml_morph.get('dep', 'unknown'))})"
                    })
                    seen_ml_indices.add(ml_idx) # Mark this ML position as handled
                    # Note: We do NOT 'continue' here because the paired EL word might still be valid for the list below

            """ Content Morphemes """
            # Only check this EL word if we haven't already marked it valid
            if el_idx not in seen_el_indices:
                if el_morph['type'] == 'content':
                    valid_el_words.append({
                        'word': el_morph['text'],
                        'pos': el_morph['upos'],
                        'ml_pos': ml_morph.get('upos', ml_morph.get('upos', 'UNKNOWN')),
                        'compatible': self._check_pos_compatibility(ml_morph, el_morph)
                    })
                    seen_el_indices.add(el_idx) # Mark this EL position as handled

        simple_el_words = [item['word'] for item in valid_el_words]
        simple_ml_words = [item['word'] for item in ml_frame_words]

        return simple_el_words, simple_ml_words, valid_el_words, ml_frame_words

    def _check_pos_compatibility(self, ml_morph, el_morph):
        """ Check if POS tags are compatible for switching """

        ml_pos = ml_morph.get('upos', ml_morph.get('upos', ''))
        el_pos = el_morph.get('upos', '')

        if self.debug:
            print(f"DEBUG POS CHECK: ML POS: {ml_pos}, EL POS: {el_pos}")

        if ml_pos == el_pos:
            return True
        if ml_pos in ['NOUN', 'PROPN'] and el_pos in ['NOUN', 'PROPN']:
            return True
    
        if ml_pos == 'VERB' and el_pos == 'VERB':
            return True

        return False

def create_mlf_alignment(src, tgt, alignment, matrix_lang, embedded_lang, lang1, lang2, 
                        example_ml, example_el, example_cs, mlf_analyzer):
    
    SYSTEM_PROMPT = (f"You are a Bilingual Hindi-English speaker. "
                    f"Generate code-mixed sentences where {matrix_lang} provides "
                    f"the grammatical frame (word order, case markers, auxiliaries, tense) "
                    f"and {embedded_lang} provides content words (nouns, verbs, adjectives).\n "
                    f"Choose any valid switching points based on the embedded language words provided, "
                    f"ensure that the final sentence is natural and fluent.\n"
                    f"You do not have to use all the embedded language words provided.\n"
                    f"Write all embedded language words in roman script, e.g. 'kitab' not 'किताब'.\n"
                    f"Use only romanized script for non-English words. Do not use devanagari script.\n"
                    f"Write Hindi words phonetically (e.g. write 'kitab' not 'किताब').\n"
                    f"IMPORTANT OUTPUT RULES:\n"
                    f"1. Output ONLY the code-mixed sentence.\n"
                    f"2. Do NOT provide translations, notes, or explanations.\n"
                    f"3. Do NOT use markdown bolding (**) or italics (*).\n"
                    f"4. Do NOT label the output (e.g. don't write 'Output:').\n"
                    f"5. Do NOT add new lines before or after the sentence.")
        
    data_packet = []

    ml_is_src = (matrix_lang.lower() == lang1.lower())

    for src_sent, tgt_sent, align_text in zip(src, tgt, alignment):
        ml_sent = src_sent if ml_is_src else tgt_sent
        el_sent = tgt_sent if ml_is_src else src_sent
        ml_lang_code = lang1 if ml_is_src else lang2
        el_lang_code = lang2 if ml_is_src else lang1

        # Get MLF-valid switching points using full parsing
        valid_el_words, ml_frame_words, detailed_el, detailed_ml = mlf_analyzer.get_valid_mlf_switches(
            ml_sent, el_sent, align_text, ml_lang_code, el_lang_code
        )

        if "gemma" in args.model_id.lower():

            prompt_messages = [
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n"
                                        f"Matrix Language ({matrix_lang}): {example_ml}\n"
                                        f"Embedded Language ({embedded_lang}) content words to use: ['goodness', 'reward']"},
                {"role": "assistant", "content": example_cs},
                {"role": "user", "content": f"Matrix Language ({matrix_lang}): {ml_sent}\n"
                                        f"Embedded Language ({embedded_lang}) content words to use: {valid_el_words}"}
            ]

        else:
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Matrix Language ({matrix_lang}): {example_ml}\n"
                                        f"Embedded Language ({embedded_lang}) content words to use: ['goodness', 'reward']"},
                {"role": "assistant", "content": example_cs},
                {"role": "user", "content": f"Matrix Language ({matrix_lang}): {ml_sent}\n"
                                        f"Embedded Language ({embedded_lang}) content words to use: {valid_el_words}"}
            ]

        data_packet.append({
            "prompt_messages": prompt_messages,
            "meta": {
                "ml_sent": ml_sent,
                "el_sent": el_sent,
                "alignment": align_text,
                "el_simple": valid_el_words,
                "ml_simple": ml_frame_words
            }
        })

    return data_packet

def create_baseline(src, tgt, lang1, lang2, example_l1, example_l2, example_cs):
    
    src_prompts = []
    tgt_prompts = []
    SYSTEM_SRC_PROMPT = (f"You are a Bilingual {lang1} {lang2} speaker, "
                         f"you will help translate these {lang1} sentence "
                         f"into a code mixed sentence with romanized {lang2} and {lang1}"
                         f"Use romanized script for non-English words. Do not use devanagari script."
                         f"Write Hindi words phonetically (e.g. write 'kitab' not 'किताब')"
                         f"IMPORTANT OUTPUT RULES:\n"
                         f"1. Output ONLY the code-mixed sentence.\n"
                         f"2. Do NOT provide translations, notes, or explanations.\n"
                         f"3. Do NOT use markdown bolding (**) or italics (*).\n"
                         f"4. Do NOT label the output (e.g. don't write 'Output:').\n"
                         f"5. Do NOT add new lines before or after the sentence."
                        )

    SYSTEM_TGT_PROMPT = (f"You are a Bilingual {lang1} {lang2} speaker, "
                         f"you will help translate these {lang2} sentence "
                         f"into a code mixed sentence with romanized {lang2} and {lang1}"
                         f"Use romanized script for non-English words. Do not use devanagari script."
                         f"Write Hindi words phonetically (e.g. write 'kitab' not 'किताब')"
                         f"IMPORTANT OUTPUT RULES:\n"
                         f"1. Output ONLY the code-mixed sentence.\n"
                         f"2. Do NOT provide translations, notes, or explanations.\n"
                         f"3. Do NOT use markdown bolding (**) or italics (*).\n"
                         f"4. Do NOT label the output (e.g. don't write 'Output:').\n"
                         f"5. Do NOT add new lines before or after the sentence."
                        )
    
    if "gemma" in args.model_id.lower():
            
        for src_sent, tgt_sent in zip(src, tgt):
            src_prompt_msgs = [
                {"role": "user", "content": SYSTEM_SRC_PROMPT + "\n\nExample:\n" + example_l1},
                {"role": "assistant", "content": example_cs},
                {"role": "user", "content": src_sent},
            ]

            tgt_prompt_msgs = [
                {"role": "user", "content": SYSTEM_TGT_PROMPT + "\n\nExample:\n" + example_l1},
                {"role": "assistant", "content": example_cs},
                {"role": "user", "content": tgt_sent},
            ]

            src_prompts.append({
                "prompt_messages": src_prompt_msgs,
                "meta": {
                    "src_sent": src_sent,
                    "tgt_sent": tgt_sent,
                    }
                })

            tgt_prompts.append({
                "prompt_messages": tgt_prompt_msgs,
                "meta": {
                    "src_sent": src_sent,
                    "tgt_sent": tgt_sent,
                    }
                })
    
    else:
        for src_sent, tgt_sent in zip(src, tgt):
            src_prompt_msgs = [
                {"role": "system", "content": SYSTEM_SRC_PROMPT},
                {"role": "user", "content": example_l1},
                {"role": "assistant", "content": example_cs},
                {"role": "user", "content": src_sent},
            ]

            tgt_prompt_msgs = [
                {"role": "system", "content": SYSTEM_TGT_PROMPT},
                {"role": "user", "content": example_l2},
                {"role": "assistant", "content": example_cs},
                {"role": "user", "content": tgt_sent},
            ]
        
            src_prompts.append({
                "prompt_messages": src_prompt_msgs,
                "meta": {
                    "src_sent": src_sent,
                    "tgt_sent": tgt_sent,
                    }
            })

            tgt_prompts.append({
                "prompt_messages": tgt_prompt_msgs,
                "meta": {
                    "src_sent": src_sent,
                    "tgt_sent": tgt_sent,
                    }
            })

    return src_prompts, tgt_prompts

"""ECT Alignment Logic"""

def get_valid_ect_switches(alignment_pairs):
    valid_pairs = []
    for i in range(len(alignment_pairs)):
        valid = True
        for j in range(len(alignment_pairs)):
            ai, bi = alignment_pairs[i]
            aj, bj = alignment_pairs[j]
            if (ai < aj and bi > bj) or (ai > aj and bi < bj):
                valid = False
                break
        if valid:
            valid_pairs.append(alignment_pairs[i])
    return valid_pairs


def create_ect_alignment(src, tgt, alignment, lang1, lang2, example_l1, example_l2, example_cs, example_words_l1, example_words_l2):

    SYSTEM_SRC_PROMPT = (f"You are a Bilingual {lang1} {lang2} speaker, "
                         f"you will help translate these {lang1} sentence "
                         f"into a code mixed sentence with romanized {lang2} "
                         f"and {lang1} with specific key words that we want to appear"
                         f"Use romanized script for non-English words. Do not use devanagari script."
                         f"Write Hindi words phonetically (e.g. write 'kitab' not 'किताब')"
                         f"IMPORTANT OUTPUT RULES:\n"
                         f"1. Output ONLY the code-mixed sentence.\n"
                         f"2. Do NOT provide translations, notes, or explanations.\n"
                         f"3. Do NOT use markdown bolding (**) or italics (*).\n"
                         f"4. Do NOT label the output (e.g. don't write 'Output:').\n"
                         f"5. Do NOT add new lines before or after the sentence."
                        )

    SYSTEM_TGT_PROMPT = (f"You are a Bilingual {lang1} {lang2} speaker, "
                         f"you will help translate these {lang2} sentence "
                         f"into a code mixed sentence with romanized {lang2} "
                         f"and {lang1} with specific key words that we want to appear"
                         f"Choose any valid switching points based on the key words provided, "
                         f"ensure that the final sentence is natural and fluent. "
                         f"You do not have to use all the key words provided."
                         f"Use romanized script for non-English words. Do not use devanagari script."
                         f"Write Hindi words phonetically (e.g. write 'kitab' not 'किताब')"
                         f"IMPORTANT OUTPUT RULES:\n"
                         f"1. Output ONLY the code-mixed sentence.\n"
                         f"2. Do NOT provide translations, notes, or explanations.\n"
                         f"3. Do NOT use markdown bolding (**) or italics (*).\n"
                         f"4. Do NOT label the output (e.g. don't write 'Output:').\n"
                         f"5. Do NOT add new lines before or after the sentence."
                        )
                        
    src_prompts = []
    tgt_prompts = []
    for src_sent, tgt_sent, align_text in zip(src,tgt,alignment):
        align_pairs = [tuple(map(int, pair.split('-'))) for pair in align_text.strip().split()]
        switching_locations = get_valid_ect_switches(align_pairs)
        src_words = src_sent.replace('\n', ' ').strip().lower().split()
        tgt_words = tgt_sent.replace('\n', ' ').strip().lower().split()
        valid_src_words = []
        valid_tgt_words = []
        for src_point, tgt_point in switching_locations:
            try:
                valid_tgt_words.append(tgt_words[tgt_point]), valid_src_words.append(src_words[src_point])
            except:
                continue
        
        if "gemma" in args.model_id.lower():

            src_prompt_msgs = [
                    {"role": "user", "content": f"{SYSTEM_SRC_PROMPT}\n"
                                                f"{example_l1}\n words wanted: {example_words_l1}"},
                    {"role": "assistant", "content": example_cs},
                    {"role": "user", "content": f"{src_sent}\n words wanted: {list(set(valid_src_words))}"},
                ]

            src_prompts.append({
            "prompt_messages": src_prompt_msgs,
            "meta": {
                "src_sent": src_sent,
                "tgt_sent": tgt_sent,
                "alignment": align_text,
                "src_words": valid_src_words,
                "tgt_words": valid_tgt_words
                }
            })
            
            tgt_prompt_msgs = [
                    {"role": "user", "content": f"{SYSTEM_TGT_PROMPT}\n"
                                                f"{example_l2}\n words wanted: {example_words_l2}"},
                    {"role": "assistant", "content": example_cs},
                    {"role": "user", "content": f"{tgt_sent}\n words wanted: {list(set(valid_tgt_words))}"},
                ]
            

            tgt_prompts.append({
            "prompt_messages": tgt_prompt_msgs,
            "meta": {
                "src_sent": src_sent,
                "tgt_sent": tgt_sent,
                "alignment": align_text,
                "src_words": valid_src_words,
                "tgt_words": valid_tgt_words
                }
            })

        else:
                
            src_prompt_msgs = [
                    {"role": "system", "content": SYSTEM_SRC_PROMPT},
                    {"role": "user", "content": f"{example_l1}\n words wanted: {example_words_l1}"},
                    {"role": "assistant", "content": example_cs},
                    {"role": "user", "content": f"{src_sent}\n words wanted: {list(set(valid_src_words))}"},
                ]
            
            src_prompts.append({
            "prompt_messages": src_prompt_msgs,
            "meta": {
                "src_sent": src_sent,
                "tgt_sent": tgt_sent,
                "alignment": align_text,
                "src_words": valid_src_words,
                "tgt_words": valid_tgt_words
                }
            })

            tgt_prompt_msgs = [
                    {"role": "system", "content": SYSTEM_TGT_PROMPT},
                    {"role": "user", "content": f"{example_l2}\n words wanted: {example_words_l2}"},
                    {"role": "assistant", "content": example_cs},
                    {"role": "user", "content": f"{tgt_sent}\n words wanted: {list(set(valid_tgt_words))}"},
                ]

            tgt_prompts.append({
            "prompt_messages": tgt_prompt_msgs,
            "meta": {
                "src_sent": src_sent,
                "tgt_sent": tgt_sent,
                "alignment": align_text,
                "src_words": valid_src_words,
                "tgt_words": valid_tgt_words
                }
            })

    return src_prompts, tgt_prompts

def get_outputs(input_list, terminators):
    """Generate outputs from model"""
    generated_texts = []
    all_scores = []
    model_id = args.model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    """bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True)"""
    
    if "Ministral-3" in args.model_id:
        print(">> DETECTED MINISTRAL 3: Using Mistral3ForConditionalGeneration class")
        model = Mistral3ForConditionalGeneration.from_pretrained(
            args.model_id,
            #quantization_config=bnb_config,
            device_map="auto",
            dtype="auto",
            trust_remote_code=True
        )
    else:
        print(f">> DETECTED OTHER MODEL: Using AutoModelForCausalLM class")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            #quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

    print(f"Generating outputs for {len(input_list)} prompts...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [args.lang1, args.mode]
    parts.append(timestamp)
    if "gemma" in args.model_id:
        parts.append("gemma")
    elif "mistral" in args.model_id:
        parts.append("mistral")
    else: 
        parts.append("llama")

    log_filename = "_".join(parts) + ".jsonl"
    open(log_filename, 'w').close()
    
    # Iterate through inputs individually to handle scores cleanly
    for i, item in enumerate(tqdm(input_list)):

        prompt_messages = item['prompt_messages']
        meta = item['meta']

        prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        
        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[1]

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            return_dict_in_generate=True,
            output_scores=True, # Critical for getting logits
            eos_token_id=terminators,
            #do_sample=True,
            #temperature=0.6,
            #top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Compute transition scores (log probabilities)
        # normalize_logits=True applies LogSoftmax
        
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        # Extract generated tokens (excluding input prompt)
        generated_tokens = outputs.sequences[:, input_length:]
        
        # Decode full text for main CSV
        text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        if args.mode == 'matrix':
            # Log MLF metadata
            log_entry = {
                "index": i,
                "original_ml_sentence": meta['ml_sent'],
                "original_el_sentence": meta['el_sent'],
                "alignment_str": meta['alignment'],
                "prompt_text": prompt_str,
                "extracted_el_words": meta['el_simple'],  # The valid words you found
                "extracted_ml_words": meta['ml_simple'],  # The system morphemes
                "generated_output": text # Result from model.generate
            }
        
        elif args.mode == 'equivalence':
            log_entry = {
                "index": i,
                "original_src_sentence": meta['src_sent'],
                "original_tgt_sentence": meta['tgt_sent'],
                "alignment_str": meta['alignment'],
                "prompt_text": prompt_str,
                "extracted_src_words": meta['src_words'], 
                "extracted_tgt_words": meta['tgt_words'], 
                "generated_output": text # 
            }

        elif args.mode == 'baseline':
            log_entry = {
                "index": i,
                "original_src_sentence": meta['src_sent'],
                "original_tgt_sentence": meta['tgt_sent'],
                "prompt_text": prompt_str,
                "generated_output": text #
            }
            
        log_to_file(log_filename, log_entry)

        generated_texts.append(text)

        # Process individual token scores
        for tok_idx, (token_id, score) in enumerate(zip(generated_tokens[0], transition_scores[0])):
            # Stop processing if we hit a terminator to avoid logging padding scores
            if token_id in terminators:
                break
                
            score_val = score.cpu().numpy() # Log probability
            prob_val = np.exp(score_val)   # Probability
            token_text = tokenizer.decode(token_id)
            
            all_scores.append({
                "prompt_id": i,
                "token_index": tok_idx,
                "token_text": token_text,
                "token_id": token_id.cpu().item(),
                "log_prob": score_val,
                "probability": prob_val
            })

    return generated_texts, all_scores

def main(args):
    lang_dict = {
        'en': 'English',
        'hi': 'Hindi',
    }

    example_dict = {
        'en': 'The reward of goodness shall be nothing but goodness',
        'hi': 'अच्छाई का पुरस्कार अच्छाई के अलावा कुछ नहीं होगा',
    }

    example_cs_dict = {
    'cs': 'goodness ka reward keval goodness hee hoga.'
    }

    example_words_dict = {
        'en': ['reward'],
        'hi': ['होगा'],
    }

    lang1 = args.lang1
    lang2 = args.lang2

    example_words_l1 = example_words_dict[lang1]
    example_words_l2 = example_words_dict[lang2]
   
    matrix_lang_code = args.matrix_lang
    embedded_lang_code = lang2 if matrix_lang_code == lang1 else lang1
    
    example_l1 = example_dict[lang1]
    example_l2 = example_dict[lang2]
    example_cs = example_cs_dict['cs']
    
    matrix_lang = lang_dict[matrix_lang_code]
    embedded_lang = lang_dict[embedded_lang_code]
    
    example_ml = example_l2 if matrix_lang_code == lang2 else example_l1
    example_el = example_l1 if matrix_lang_code == lang2 else example_l2
    
    # Load model
    print(f"\nLoading LLM: {args.model_id}")
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    terminators = [tokenizer.eos_token_id]

    if args.eot_token:
        terminators.append(tokenizer.convert_tokens_to_ids(args.eot_token))
    

    print("\nLoading data files...")
    src = read_file(args.src) if args.src else None
    tgt = read_file(args.tgt) if args.tgt else None
    src_translated = read_file(args.src_translated) if args.src_translated else None
    tgt_translated = read_file(args.tgt_translated) if args.tgt_translated else None
    gold_align = read_file(args.gold_align) if args.gold_align else None
    
    data = {}
    if src: data['src'] = src
    if tgt: data['tgt'] = tgt
    if src_translated: data['src_translated'] = src_translated
    if tgt_translated: data['tgt_translated'] = tgt_translated

    if args.mode =='matrix':

        # Initialize MLF analyzer with Stanza
        print("="*60)
        print("Initializing Unconstrained MLF Analyzer with Stanza")
        print("="*60)
        mlf_analyzer = MLFAnalyser(
            lang1=lang1, 
            lang2=lang2,
            download_models=args.download_stanza
        )

        # MLF with Gold alignment
        print("\n" + "="*60)
        print(f"Matrix Language Frame Generation")
        print(f"  Matrix Language: {matrix_lang}")
        print(f"  Embedded Language: {embedded_lang}")
        print("="*60)
        if src and tgt and gold_align:
            mlf_gold = create_mlf_alignment(src, tgt, gold_align, matrix_lang_code, embedded_lang_code,
                                            lang1, lang2, example_ml, example_el, 
                                            example_cs, mlf_analyzer)
            if mlf_gold:
                print(f"Generating {len(mlf_gold)} MLF-compliant outputs...")
                generated_texts, score_data = get_outputs(mlf_gold, terminators)
                data['mlf_gold'] = generated_texts
                pd.DataFrame(data).to_csv(args.output, index=False)
                print(f"✓ Saved to {args.output}")

                score_output_path = args.output.replace('.csv', '_scores.csv')
                if score_output_path == args.output: # Handle case where .csv wasn't in extension
                    score_output_path = args.output + '_scores.csv'
                    
                df_scores = pd.DataFrame(score_data)
                df_scores.to_csv(score_output_path, index=False)
                print(f"✓ Saved detailed token scores to {score_output_path}")

    elif args.mode == 'equivalence':
        
        print("\n" + "="*60)
        print(f"Equivalence Constraint Generation")
        print("="*60)

        ect_src, ect_tgt = create_ect_alignment(src, tgt, gold_align, lang1, lang2, 
            example_l1, example_l2, example_cs,
            example_words_dict[lang1], example_words_dict[lang2])
        
        if ect_src and ect_tgt:
            print(f"\n--- Processing Direction A: {lang1} Base ---")
            texts_src, scores_src = get_outputs(ect_src, terminators)
            
            # 3. Run Inference on List B (Target Base / Hindi Base)
            print(f"\n--- Processing Direction B: {lang2} Base ---")
            texts_tgt, scores_tgt = get_outputs(ect_tgt, terminators)

            # 4. Create Wide-Format DataFrame (Side-by-Side)
            # We assume lists are perfectly aligned by index
            df_out = pd.DataFrame({
                'src_original': [m['meta']['src_sent'] for m in ect_src],
                'tgt_original': [m['meta']['tgt_sent'] for m in ect_tgt],
                
                # Direction A Columns
                f'output_{lang1}_base': texts_src,
                f'words_{lang1}_base': [m['meta']['src_words'] for m in ect_src],
                
                # Direction B Columns
                f'output_{lang2}_base': texts_tgt,
                f'words_{lang2}_base': [m['meta']['tgt_words'] for m in ect_tgt]
            })

            # Save Main CSV
            df_out.to_csv(args.output, index=False)
            print(f"Saved results to {args.output}")

            # Save Scores (Concatenated long format)
            all_scores = scores_src + scores_tgt
            score_path = args.output.replace('.csv', '_scores.csv')
            pd.DataFrame(all_scores).to_csv(score_path, index=False)
            print(f"Saved detailed scores to {score_path}")

    elif args.mode == 'baseline':
        print("\n" + "="*60)
        print(f"Baseline Generation")
        print("="*60)

        baseline_src, baseline_tgt = create_baseline(
            src, tgt, lang1, lang2, example_l1, example_l2, example_cs
        )
        
        if baseline_src and baseline_tgt:
            print(f"Generating {len(baseline_src)} baseline outputs for source language...")
            src_texts, src_scores = get_outputs(baseline_src, terminators)
            
            print(f"Generating {len(baseline_tgt)} baseline outputs for target language...")
            tgt_texts, tgt_scores = get_outputs(baseline_tgt, terminators)

            # Extract metadata to DataFrame
            df_out = pd.DataFrame({
                'src_original': [m['meta']['src_sent'] for m in baseline_src],
                'tgt_original': [m['meta']['tgt_sent'] for m in baseline_tgt],
                
                # Direction A Columns
                f'output_{lang1}_base': src_texts,
                
                # Direction B Columns
                f'output_{lang2}_base': tgt_texts,
            })
        
            # Save Main CSV
            df_out.to_csv(args.output, index=False)
            print(f"Saved results to {args.output}")

            score_data = src_scores + tgt_scores
            df_out.to_csv(args.output.replace('.csv', '_scores.csv'), index=False)
            print(f"Saved to {args.output.replace('.csv', '_scores.csv')}")

            score_path = args.output.replace('.csv', '_scores.csv')
            pd.DataFrame(score_data).to_csv(score_path, index=False)

if __name__ == "__main__":
    args = init()
    main(args)