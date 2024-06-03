import cProfile
import pstats
import io
import gradio as gr
import stanza
from deep_translator import GoogleTranslator
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
from wordfreq import word_frequency
import difflib
import time
import pandas as pd
import re
import rapidfuzz



# Initialize Stanza pipelines (only once) with specific components
stanza.download('en')
stanza.download('es')
nlp_native = stanza.Pipeline('en', processors='tokenize,lemma,pos')
nlp_target = stanza.Pipeline('es', processors='tokenize,lemma,pos,ner')

# Initialize the Snowball Stemmer
stemmer = SnowballStemmer("spanish")

# Load the TSV file into a DataFrame, skipping bad lines
cognet_df = pd.read_csv('CogNet-v2.0.tsv', sep='\t', header=None,
                        names=['concept_id', 'lang1', 'word1', 'lang2', 'word2', 'translit1', 'translit2'],
                        on_bad_lines='skip')

# Filter for Spanish-English cognates
cognet_sp_en = cognet_df[((cognet_df['lang1'] == 'spa') & (cognet_df['lang2'] == 'eng')) |
                         ((cognet_df['lang1'] == 'eng') & (cognet_df['lang2'] == 'spa'))]

# Variables to hold state
state = {
    'paragraphs': [],
    'current_paragraph_index': 0,
    'known_words': [],
    'unknown_words': [],
    'validated_translations': [],
    'word_count': defaultdict(int),
    'all_final_unknown_words': [],
    'all_cognate_pairs': {},
    'final_unknown_words_dict': defaultdict(set),
    'original_word_mapping': {},
    'native_language': '',
    'target_language': '',
    'level': '',
    'final_unknown_word_counts': defaultdict(int),  # New field for storing unknown word counts
    'nlp_cache': {},  # Cache for NLP processing results
    'frequency_cache': {},  # Cache for word frequency calculations
    'ner_cache': {}  # Cache for NER results
}

frequency_thresholds = {
    'A1': 0.0001,
    'A2': 0.00001,
    'B1': 0.000001,
    'B2': 0.0000001,
    'C1': 0.00000001,
    'C2': 0.0
}

translation_cache = {}

def initialize_variables():
    global state
    state = {
        'paragraphs': [],
        'current_paragraph_index': 0,
        'known_words': [],
        'unknown_words': [],
        'validated_translations': [],
        'word_count': defaultdict(int),
        'all_final_unknown_words': [],
        'all_cognate_pairs': {},
        'final_unknown_words_dict': defaultdict(set),
        'original_word_mapping': {},
        'native_language': '',
        'target_language': '',
        'level': '',
        'final_unknown_word_counts': defaultdict(int),  # Reset the count of unknown words
        'nlp_cache': {},  # Reset cache for NLP processing results
        'frequency_cache': {},  # Reset cache for word frequency calculations
        'ner_cache': {},  # Reset cache for NER results
        'merged_paragraphs': []  # New state variable to keep track of merged paragraphs
    }


# Function to identify cognates
def find_cognates(spanish_words, english_words, cognet_df, similarity_threshold=0.6):
    cognates = []
    for sp_word in spanish_words:
        matches = cognet_df[(cognet_df['word1'] == sp_word) | (cognet_df['word2'] == sp_word)]
        for index, row in matches.iterrows():
            if row['lang1'] == 'spa' and row['lang2'] == 'eng' and row['word2'] in english_words:
                en_word = row['word2']
            elif row['lang1'] == 'eng' and row['lang2'] == 'spa' and row['word1'] in english_words:
                en_word = row['word1']
            else:
                continue
            similarity = rapidfuzz.fuzz.ratio(sp_word, en_word)
            if similarity >= similarity_threshold * 100:
                cognates.append((sp_word, en_word))
    return cognates

# Safely translate a sentence with retry logic
def safe_translate(sentences, src, dest, retries=3):
    translations = []
    for sentence in sentences:
        if sentence in translation_cache:
            translations.append(translation_cache[sentence])
        else:
            for _ in range(retries):
                try:
                    translation = GoogleTranslator(source=src, target=dest).translate(sentence)
                    if translation:
                        translations.append(translation)
                        translation_cache[sentence] = translation
                        break
                except Exception as e:
                    print(f"Error translating sentence '{sentence}': {e}")
                    time.sleep(1)
            else:
                translations.append("Translation not available")
    return translations

# Batch translation
def batch_translate(sentences, src, dest):
    batch_text = "\n".join(sentences)
    translation = GoogleTranslator(source=src, target=dest).translate(batch_text)
    return translation.split("\n")

# Function to check morphological similarity
def is_similar_morphology(word1, word2, threshold):
    stem1 = word1['stem']
    stem2 = word2['stem']
    if stem1 == stem2:
        return True
    if (stem1 in stem2 or stem2 in stem1) and (word2['frequency'] < threshold):
        return True
    lemma1 = word1['lemma']
    lemma2 = word2['lemma']
    if lemma1 == lemma2:
        return True
    if (lemma1 in lemma2 or lemma2 in lemma1) and (word2['frequency'] < threshold):
        return True
    return False

# Preprocess text (tokenize, lemmatize, POS tagging)
def batch_preprocess_text(paragraphs, nlp, target_language, use_cache=True):
    batch_text = "\n\n".join(paragraphs)
    if use_cache and batch_text in state['nlp_cache']:
        return state['nlp_cache'][batch_text]

    doc = nlp(batch_text)
    sentences = [sentence.text for sentence in doc.sentences]
    words = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if len(word.text) > 1:  # Ensure we are processing only whole words
                if word.text in state['frequency_cache']:
                    frequency = state['frequency_cache'][word.text]
                else:
                    frequency = word_frequency(word.text, target_language)
                    state['frequency_cache'][word.text] = frequency
                words.append({
                    'text': word.text,
                    'lemma': word.lemma,
                    'pos': word.upos,
                    'frequency': frequency,
                    'stem': stemmer.stem(word.text)
                })

    if use_cache:
        state['nlp_cache'][batch_text] = (sentences, words)
    return sentences, words

# Perform Named Entity Recognition (NER)
def perform_ner(text, nlp):
    if text in state['ner_cache']:
        return state['ner_cache'][text]

    doc = nlp(text)
    entities = []
    for sentence in doc.sentences:
        for entity in sentence.ents:
            entities.append(entity.text.lower())  # Normalize to lowercase
    state['ner_cache'][text] = entities
    print(f"Identified Entities: {entities}")  # Debugging: Print identified entities
    return entities

# Validate translation with context
def validate_translation_in_context(translation, original_sentences, translated_sentences, spanish_pos):
    for orig_sent, trans_sent in zip(original_sentences, translated_sentences):
        if trans_sent in state['nlp_cache']:
            doc = state['nlp_cache'][trans_sent]
        else:
            doc = nlp_native(trans_sent)
            state['nlp_cache'][trans_sent] = doc

        if orig_sent in state['nlp_cache']:
            orig_doc = state['nlp_cache'][orig_sent]
        else:
            orig_doc = nlp_target(orig_sent)
            state['nlp_cache'][orig_sent] = orig_doc

        # Initial Check: Direct match
        for word in doc.sentences[0].words:
            if word.text.lower() == translation.lower():
                return word.text

        # Enhanced Contextual Comparison
        original_words = [word.text for word in orig_doc.sentences[0].words]
        translated_words = [word.text for word in doc.sentences[0].words]

        # Similarity Check: Find the most similar word
        words_in_trans_sent = [word.text for word in doc.sentences[0].words]
        most_similar = rapidfuzz.process.extractOne(translation, words_in_trans_sent, scorer=rapidfuzz.fuzz.ratio, score_cutoff=70)
        if most_similar:
            similar_word = most_similar[0]
            for word in doc.sentences[0].words:
                if word.text == similar_word and word.upos == spanish_pos:
                    return f"{translation}/{similar_word}"  # Return both the individual translation and the most similar word with the same POS tag

        # POS Tag Check: Find a word with the same POS tag as the Spanish word
        pos_matches = [word.text for word in doc.sentences[0].words if word.upos == spanish_pos]
        if len(pos_matches) == 1:
            return f"{translation}/{pos_matches[0]}"  # Return both the individual translation and the POS matching word if there's only one match
        elif len(pos_matches) > 1:
            return translation  # Stick to the individual translation if multiple POS matches are found

    # If no word with the same POS tag is found
    return translation

# Profile Decorator
def profile_func(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(10)  # Print only the top 10 most expensive functions
        print(s.getvalue())
        return result
    return wrapper

# Translate and process each paragraph
@profile_func
def process_paragraph(paragraphs, input_unknown_words, known_words, unknown_words, validated_translations, word_count):
    if not paragraphs:
        return [], [], [], [], {}

    sentences, words = batch_preprocess_text(paragraphs, nlp_target, state['target_language'])
    entities = perform_ner("\n\n".join(paragraphs), nlp_target)  # Ensure NER is performed and printed
    translated_sentences = batch_translate(sentences, state['target_language'], state['native_language'])
    threshold = frequency_thresholds[state['level']]
    spanish_words = [word['text'].lower() for word in words]
    english_words = batch_translate([word['text'] for word in words], state['target_language'], state['native_language'])
    cognates = find_cognates(spanish_words, english_words, cognet_sp_en)
    cognate_pairs = {sp: en for sp, en in cognates}

    current_paragraph_unknown_words = defaultdict(set)
    for word in words:
        word_text = word['text'].lower()
        state['original_word_mapping'][word_text] = word['text']
        if word_text in entities or word['pos'] == 'PUNCT':
            print(f"Known Word (Proper Name or PUNCT): {word_text}")  # Debugging: Print known words (proper names or punctuation)
            known_words.append(word)
        elif word['frequency'] >= threshold or word_text in cognate_pairs:
            known_words.append(word)
        else:
            print(f"Unknown Word Identified: {word_text}")  # Debugging: Print unknown words
            state['final_unknown_words_dict'][word_text].add(word['lemma'])
            current_paragraph_unknown_words[word_text].add(word['lemma'])

    # Process only relevant input unknown words
    input_unknown_word_details = []
    for unknown_word in input_unknown_words:
        if unknown_word:
            print(f"Processing input unknown word: {unknown_word}")  # Debugging: Print input unknown words
            processed_words = batch_preprocess_text([unknown_word], nlp_target, state['target_language'], use_cache=False)[1]
            if processed_words:
                processed_word = processed_words[0]
                word_text = processed_word['text'].lower()
                input_unknown_word_details.append(processed_word)
                if word_text in spanish_words:
                    state['final_unknown_words_dict'][word_text].add(processed_word['lemma'])
                    current_paragraph_unknown_words[word_text].add(processed_word['lemma'])

    for word in words:
        for unknown_word in input_unknown_word_details:
            if is_similar_morphology(word, unknown_word, threshold):
                state['final_unknown_words_dict'][word['text'].lower()].add(word['lemma'])
                current_paragraph_unknown_words[word['text'].lower()].add(word['lemma'])

    for word_text, lemmas in state['final_unknown_words_dict'].items():
        for word in words:
            if is_similar_morphology({'text': word_text, 'lemma': next(iter(lemmas)), 'stem': stemmer.stem(word_text)}, word, threshold):
                current_paragraph_unknown_words[word['text'].lower()].add(word['lemma'])

    final_unknown_words = []
    for word_text, lemmas in current_paragraph_unknown_words.items():
        final_unknown_words.append({
            'text': word_text,
            'lemma': next(iter(lemmas)),
            'pos': next((word['pos'] for word in words if word['text'].lower() == word_text), 'UNKNOWN'),
            'frequency': next((word['frequency'] for word in words if word['text'].lower() == word_text), 0.0),
            'stem': stemmer.stem(word_text)
        })

    def map_words_to_sentences(sentences, words):
        sentence_word_map = {}
        for i, sentence in enumerate(sentences):
            for word in words:
                if word['text'].lower() in sentence.lower():
                    sentence_word_map[word['text'].lower()] = (sentence, i)
        return sentence_word_map

    sentence_word_map = map_words_to_sentences(sentences, final_unknown_words)
    for word in final_unknown_words:
        if word['text'].lower() not in sentence_word_map:
            print(f"Word not found in sentence_word_map: {word['text'].lower()}")  # Debug print
            continue
        if word['text'] in translation_cache:
            translation = translation_cache[word['text']]
        else:
            translation = GoogleTranslator(source=state['target_language'], target=state['native_language']).translate(word['text'])
            translation_cache[word['text']] = translation
        validated_translation = validate_translation_in_context(
            translation,
            sentences,
            translated_sentences,
            word['pos']
        )
        word['translation'] = validated_translation  # Add validated translation directly to the word dictionary
        word['sentence'] = sentence_word_map[word['text'].lower()][0]
        word['translated_sentence'] = translated_sentences[sentence_word_map[word['text'].lower()][1]]
        validated_translations.append({
            'original': word['text'],
            'translation': validated_translation,
            'translated_pos': word['pos']
        })
        word_count[word['text']] += 1
        state['final_unknown_word_counts'][word['text']] += 1  # Increment the count of the unknown word

    print(f"Final unknown words: {final_unknown_words}")  # Debugging: Print final unknown words
    return sentences, translated_sentences, final_unknown_words, validated_translations, cognate_pairs

@profile_func
def start_processing(native_language, target_language, level, text):
    initialize_variables()
    state['native_language'] = native_language
    state['target_language'] = target_language
    state['level'] = level
    original_paragraphs = text.strip().split('\n')

    # Merge single-sentence paragraphs with the following paragraph
    merged_paragraphs = []
    temp_paragraph = ""
    for paragraph in original_paragraphs:
        if len(paragraph.split('. ')) <= 1:  # Check if the paragraph is a single sentence
            if temp_paragraph:
                temp_paragraph += " " + paragraph
            else:
                temp_paragraph = paragraph
        else:
            if temp_paragraph:
                merged_paragraphs.append(temp_paragraph + " " + paragraph)
                temp_paragraph = ""
            else:
                merged_paragraphs.append(paragraph)

    if temp_paragraph:  # Add any remaining paragraph
        merged_paragraphs.append(temp_paragraph)

    state['paragraphs'] = merged_paragraphs
    state['current_paragraph_index'] = 0

    return process_next_paragraph([])



def process_next_paragraph(input_unknown_words):
    global state
    if state['current_paragraph_index'] < len(state['paragraphs']):
        paragraph = state['paragraphs'][state['current_paragraph_index']].strip()
        if not paragraph:
            state['current_paragraph_index'] += 1
            return process_next_paragraph(input_unknown_words)
        sentences, translated_sentences, final_unknown_words, validated_translations, cognate_pairs = process_paragraph(
            [paragraph],
            input_unknown_words,
            state['known_words'],
            state['unknown_words'],
            state['validated_translations'],
            state['word_count']
        )
        state['all_final_unknown_words'].extend(final_unknown_words)
        state['all_cognate_pairs'].update(cognate_pairs)
        output = display_output(paragraph, final_unknown_words)
        state['current_paragraph_index'] += 1
        print(f"Paragraph {state['current_paragraph_index']} processed with input unknown words: {input_unknown_words}")  # Debugging: Print paragraph processing status
        return output
    else:
        summary = generate_summary()
        return f"All paragraphs processed<br>{summary}"

def highlighted_paragraph(paragraph, final_unknown_words, validated_translations):
    highlighted_paragraph = paragraph
    for word in final_unknown_words:
        original_word = state['original_word_mapping'].get(word['text'], word['text'])
        translation_info = next((item for item in validated_translations if item['original'] == word['text']), None)
        if translation_info:
            translation = translation_info['translation']
            highlighted_paragraph = re.sub(r'\b{}\b'.format(re.escape(original_word)), f"<b>{original_word}</b>({translation})", highlighted_paragraph, flags=re.IGNORECASE)
    print(f"Highlighted paragraph: {highlighted_paragraph}")  # Debugging: Print highlighted paragraph
    return highlighted_paragraph

def display_output(paragraph, final_unknown_words):
    highlighted_para = highlighted_paragraph(paragraph, final_unknown_words, state['validated_translations'])
    context_sentences = []
    for word in final_unknown_words:
        translation = word.get('translation', 'No translation available')
        context_sentence = f"{word['text']}: {translation}. {word.get('translated_sentence', 'No sentence available')}"
        context_sentences.append(context_sentence)

    context_output = "<br>".join(context_sentences)

    # Restore original paragraph formatting
    original_paragraphs = paragraph.split(' ')
    highlighted_original_para = highlighted_paragraph(" ".join(original_paragraphs), final_unknown_words, state['validated_translations'])

    print(f"Output for paragraph: {highlighted_para}")  # Debugging: Print output for paragraph
    return f"<p>Highlighted Text:</p><p>{highlighted_original_para}</p><p>Predicted Unknown Words In Context:</p><p>{context_output}</p>"



def generate_summary():
    summary = "Summary of Unknown Words:<br>"
    translations_dict = {}

    for word in state['final_unknown_word_counts'].keys():
        translation_info = next((item for item in state['validated_translations'] if item['original'] == word), None)
        if translation_info:
            translations_dict[word] = translation_info['translation']

    for word, count in state['final_unknown_word_counts'].items():
        translation = translations_dict.get(word, 'No translation found')
        summary += f"{word}: {count} appearances, Translation: {translation}<br>"

    print(f"Summary: {summary}")  # Debugging: Print summary
    return summary

def next_paragraph(input_unknown_words):
    if isinstance(input_unknown_words, str):
        input_unknown_words = input_unknown_words.split()  # Ensure input_unknown_words is a list
    return process_next_paragraph(input_unknown_words)

def reset_interface():
    global state
    state = {
        'paragraphs': [],
        'current_paragraph_index': 0,
        'known_words': [],
        'unknown_words': [],
        'validated_translations': [],
        'word_count': defaultdict(int),
        'all_final_unknown_words': [],
        'all_cognate_pairs': {},
        'final_unknown_words_dict': defaultdict(set),
        'original_word_mapping': {},
        'native_language': '',
        'target_language': '',
        'level': '',
        'final_unknown_word_counts': defaultdict(int)  # Reset the count of unknown words
    }
    return gr.update(value=''), gr.update(value=''), gr.update(value=''), gr.update(value=''), gr.update(value=''), gr.update(value='')

# Gradio Interface
iface = gr.Blocks()

with iface:
    native_language_input = gr.Dropdown(choices=['en'], label='Native Language', value='en')
    target_language_input = gr.Dropdown(choices=['es', 'it'], label='Target Language')
    level_input = gr.Dropdown(choices=['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], label='Level')
    text_input = gr.Textbox(label='Text', lines=10)
    start_button = gr.Button('Start')
    output_area = gr.HTML()
    unknown_words_input = gr.Textbox(label='Input Unknown Words', lines=2)
    next_button = gr.Button('Next Paragraph')
    restart_button = gr.Button('Restart')

    start_button.click(start_processing, [native_language_input, target_language_input, level_input, text_input], [output_area])
    next_button.click(next_paragraph, [unknown_words_input], [output_area])
    restart_button.click(reset_interface, [], [native_language_input, target_language_input, level_input, text_input, output_area, unknown_words_input])

iface.launch(share=True, debug=True)