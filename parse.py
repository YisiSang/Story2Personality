import os, sys, json
from pathlib import Path
import csv
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
import spacy
import pandas as pd
import glob
import pickle

# spacy language corpus
NLP_PIPE = spacy.load("en_core_web_sm")

def spacy_entity(text):    
    return [[w.text, w.label_] for w in NLP_PIPE(text).ents]


def load_story_meta_data():
    movie_dict = {}
    filename = 'narrative_qa/storedScript.csv'
    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
        for i, row in enumerate(csvreader):
            if len(row) != 2:
                print('[X] Error: different length:', len(row), row)
            if i == 0:
                continue
            movie_dict[row[1]] = row[0]

    exclusive_id = [
        '39b1f44ec2639226b6025140e2a7e7eaf570409e',     # content format issue
        '492f2d56eba93816e7d0958e2ba62d36d93bc97e',     # content format issue
        '0269408ac78193a0e13f1033bec6f658b00437b3',     # content format issue
        '0bc7352d6a0e678c0d8acc57c0c1cc3466fe9ef7',     # no <b></b>
        '405df1ccf0409ea6040c2a765a0315878f991d79',     # no <b></b>
        '4112d61d9880d57229b3a49a5d04e0bc534e44a8',     # no    </b>
        '823a8b6bbbb0bbeec9c4bec1929b64a3694232d7',     # no <b></b>
        '1aae28477e771b3af008ec59ce29086a1bc66776',     # download failed?
        '3747036f950fe8f79cdaa0eb713104b9eb8af6c5',     # download failed?
        '5283fa0a6ea69f4b4224d12018bbf985a2b80543',     # download failed?
    ]

    special_id = [
        '08172f313a4053b150a6c20310b6ee58f6968b76',     # no <b></b>
        '37c11f984cb14401d85abfc20e8305ca7a472c9f',     # no <b></b>
    ]

    story_meta = dict()
    with open('narrative_qa/documents.csv','r') as fp:
        content = csv.reader(fp, delimiter=',')
        for each_line in content:
            if content.line_num == 1:
                print('[*] Columns in documents.csv:', each_line)
                continue
            # 0: 'document_id', 
            # 1: 'set', 
            # 2: 'kind', 
            # 3: 'story_url', 
            # 4: 'story_file_size', 
            # 5: 'wiki_url', 
            # 6: 'wiki_title', 
            # 7: 'story_word_count'
            # 8: 'story_start'
            # 9: 'story_end'
            
            if each_line[0] not in movie_dict:
                continue
            if each_line[0] in exclusive_id or each_line[0] in special_id:
                continue
            
            story_meta[each_line[0]] = dict()
            story_meta[each_line[0]]['movie_name']      = movie_dict[each_line[0]]
            story_meta[each_line[0]]['set']             = each_line[1]
            story_meta[each_line[0]]['kind']            = each_line[2]
            story_meta[each_line[0]]['story_url']       = each_line[3]
            story_meta[each_line[0]]['story_file_size'] = each_line[4]
            story_meta[each_line[0]]['wiki_url']        = each_line[5]
            story_meta[each_line[0]]['wiki_title']      = each_line[6]
            story_meta[each_line[0]]['story_word_count'] = each_line[7]
            story_meta[each_line[0]]['story_start']     = each_line[8]
            story_meta[each_line[0]]['story_end']       = each_line[9]

    return story_meta
    

def analyze_segmentation_signals():
    results = {}
    list_num_scenes = []
    list_num_dialogs = []

    num_verified = 0
    num_failed = 0
    num_movie = 0
    num_fade_in_success = 0

    for file_path in tqdm(glob.glob("./raw_texts/*.txt"), desc="Analyze segmentation signals"):
        num_movie += 1
        num_scenes = 0
        num_dialogs = 0

        each_id = Path(file_path).stem
        with open(file_path, 'r', encoding='cp1256') as fp:
            content = fp.read()

        soup = BeautifulSoup(content, 'html.parser')
        bs = soup.pre.find_all('b')

        sents = [
            (element.string, [ele for ele in element.next_sibling.string.split('\n') if ele.strip()]) # split the text by \n; remove space
                for element in bs if (element.next_sibling and element.next_sibling.string is not None) # select text 
        ]
        
        scene_symbol = None
        scene_found = False
        fade_in_indent = -1

        for element in bs:
            if element is None or element.string is None:
                continue
            if not scene_found and element.string.find('FADE IN') != -1: # find if there is a FADE IN in the sentence
                scene_symbol = str(element.string)[:element.string.index('FADE IN')] # count the space before FADE IN
                scene_found = True # only need the 1st FADE IN (incase there is any FADE IN in the text)

            if scene_found and element.string.startswith(scene_symbol + scene_symbol): # find if there are texts start with 2x the space before FADE IN
                dialogue_symbol = scene_symbol + scene_symbol
                break

        if scene_symbol is not None: # there is no FADE IN in the script
            pointer = 0
            if pointer != len(scene_symbol):
                while scene_symbol[pointer] == ' ' or scene_symbol[pointer] == '\t' or scene_symbol[pointer] == '\n':
                    pointer += 1
                    if pointer == len(scene_symbol):
                        break
            fade_in_indent = pointer
        
        meta_info = {}
        meta_info['id'] = each_id
        meta_info["sents"] = sents
        meta_info["scene_symbol"] = scene_symbol
        meta_info["scene_found"] = scene_found
        if scene_symbol is None:
            meta_info['fade-in symbol'] = []
        else:
            meta_info['fade-in symbol'] = [scene_symbol]
        meta_info['fade-in indent'] = fade_in_indent
        
        scene_indent_dict = {}
        indent_dict = {}
        dialog_indent_dict = {}

        for section in sents:
            if len(section[1]) == 0:
                continue
            # extract section name (character name)
            if section[0] is not None:
                section_name = re.sub(r'\t', '', section[0]).strip()
            else:
                section_name = 'NULL'
            # extract or concatenate texts (utterance)

            prev_indent = -1
            new_scene = False
            for sent in section[1]:
                pointer = 0
                if pointer == len(sent):
                    continue
                while sent[pointer] == ' ' or sent[pointer] == '\t' or sent[pointer] == '\n':
                    pointer += 1
                    if pointer == len(sent):
                        break
                if pointer == len(sent):
                    continue

                indent = pointer
                if indent not in indent_dict:
                    indent_dict[indent] = 1
                else:
                    indent_dict[indent] += 1
        
                if fade_in_indent != -1:
                    if indent > fade_in_indent:
                        if indent not in dialog_indent_dict:
                            dialog_indent_dict[indent] = 1
                        else:
                            dialog_indent_dict[indent] += 1
                        num_dialogs += 1
                    else:
                        if indent not in scene_indent_dict:
                            scene_indent_dict[indent] = 1
                        else:
                            scene_indent_dict[indent] += 1
                        num_scenes += 1

        if len(indent_dict) == 0:
            meta_info['type'] = 'Fail'
            meta_info['indent_dict'] = indent_dict
        elif fade_in_indent == -1:
            sorted_indents = sorted(indent_dict.items(), key=lambda x:x[0])
            num_total = 0
            for k, v in indent_dict.items():
                num_total += v
                
            scene_indent_dict = {}
            dialog_indent_dict = {}
            cur_num_scenes = 0
            adding_scene = True
            failure = False
            split_num = 0
            for idx, indent_pair in enumerate(sorted_indents):
                cur_ratio = cur_num_scenes / num_total
                next_ratio = (cur_num_scenes + indent_pair[1]) / num_total
                # a huge jump refers to failure case
                if cur_ratio <= 0.3870662796573337 - 0.13559559099176607 and next_ratio >= 0.60: # 0.3870662796573337 + 0.13559559099176607:
                    failure = True
                    break
                if adding_scene and next_ratio >= 0.6: #0.3870662796573337 + 0.13559559099176607: # split here
                    adding_scene = False
                    split_num = indent_pair[0]
                if adding_scene:
                    scene_indent_dict[indent_pair[0]] = indent_pair[1]
                    cur_num_scenes += indent_pair[1]
                else:
                    dialog_indent_dict[indent_pair[0]] = indent_pair[1]
            if not failure:
                num_verified += 1
                meta_info['type'] = 'Reorg overall'
                meta_info['indent_dict'] = indent_dict
                meta_info['scene_dict'] = scene_indent_dict
                meta_info['dialog_dict'] = dialog_indent_dict
            else:
                num_failed += 1
                meta_info['type'] = 'Fail'
                meta_info['indent_dict'] = indent_dict
        else:
            list_num_scenes.append(num_scenes)
            list_num_dialogs.append(num_dialogs)
            
            ratio_scene = num_scenes / (num_scenes + num_dialogs)
            if ratio_scene <= 0.6 and ratio_scene >= 0.3870662796573337 - 0.13559559099176607:
                num_verified += 1
                num_fade_in_success += 1
                meta_info['type'] = 'Original fade-in'
                meta_info['indent_dict'] = indent_dict
                meta_info['scene_dict'] = scene_indent_dict
                meta_info['dialog_dict'] = dialog_indent_dict
            else:
                sorted_indents = sorted(indent_dict.items(), key=lambda x:x[0])
                num_total = 0
                for k, v in indent_dict.items():
                    num_total += v

                scene_indent_dict = {}
                dialog_indent_dict = {}
                cur_num_scenes = 0
                adding_scene = True
                failure = False
                split_num = 0
                for idx, indent_pair in enumerate(sorted_indents):
                    cur_ratio = cur_num_scenes / num_total
                    next_ratio = (cur_num_scenes + indent_pair[1]) / num_total
                    # a huge jump refers to failure case
                    if cur_ratio <= 0.3870662796573337 - 0.13559559099176607 and next_ratio >= 0.6: # 0.3870662796573337 + 0.13559559099176607:
                        failure = True
                        break
                    if adding_scene and next_ratio >= 0.6: # 0.3870662796573337 + 0.13559559099176607: # split here
                        adding_scene = False
                        split_num = indent_pair[0]
                    if adding_scene:
                        scene_indent_dict[indent_pair[0]] = indent_pair[1]
                        cur_num_scenes += indent_pair[1]
                    else:
                        dialog_indent_dict[indent_pair[0]] = indent_pair[1]
                if not failure:
                    num_verified += 1
                    meta_info['type'] = 'Reorg fade-in'
                    meta_info['indent_dict'] = indent_dict
                    meta_info['scene_dict'] = scene_indent_dict
                    meta_info['dialog_dict'] = dialog_indent_dict
                else:
                    num_failed += 1
                    meta_info['type'] = 'Fail'
                    meta_info['indent_dict'] = indent_dict

        results[each_id] = meta_info
        
    print("[*] Segmentation Analysis Results")
    print("    - Success:", num_verified)
    print("    - Fail:", num_failed)
    print("    - # of movies:", num_movie)
    return results


def read_and_split_documents(split_meta):
    Path('tmp/silver').mkdir(parents=True, exist_ok=True)
    Path('tmp/by_stats').mkdir(parents=True, exist_ok=True)
    Path('tmp/bad_format').mkdir(parents=True, exist_ok=True)

    total_scenes = 0
    total_dialogs = 0
    for each_id in tqdm(split_meta.keys(), desc="Segmenting"):
        sents = split_meta[each_id]["sents"]

        if split_meta[each_id]['type'] == 'Fail':
            if len(sents) < 200:
                continue
            else:
                new_sections = list()
                for sent in sents:
                    if sent[0] is not None:
                        section_name = re.sub(r'\t', '', sent[0]).strip()
                    else:
                        section_name = re.sub(r'\t', '', '\n').strip()
                    section_texts = ''
                    for line in sent[1]:
                        section_texts += re.sub(r'\t', '', line).strip() + '\n'

                    new_sections.append(('UNK', section_name, section_texts))
        else:
            # with good format
            num_scenes = 0
            num_dialogs = 0
            scene_symbol = split_meta[each_id]["scene_symbol"]
            scene_found = split_meta[each_id]["scene_found"]

            if scene_symbol is not None: # there is no FADE IN in the script
                pointer = 0
                if pointer != len(scene_symbol):
                    while scene_symbol[pointer] == ' ' or scene_symbol[pointer] == '\t' or scene_symbol[pointer] == '\n':
                        pointer += 1
                        if pointer == len(scene_symbol):
                            break
            
            new_sections = []
            for section in sents:
                if len(section[1]) == 0:
                    continue
                # extract section name (character name)
                if section[0] is not None:
                    section_name = re.sub(r'\t', '', section[0]).strip()
                else:
                    section_name = 'NULL'

                # extract or concatenate section text (utterance)
                section_texts = ''
                prev_indent = -1
                new_scene = False
                for sent in section[1]:
                    pointer = 0
                    if pointer == len(sent):
                        continue
                    while sent[pointer] == ' ' or sent[pointer] == '\t' or sent[pointer] == '\n':
                        pointer += 1
                        if pointer == len(sent):
                            break
                    if pointer == len(sent):
                        continue
                    # indent = str(pointer)
                    indent = pointer
                    if indent in split_meta[each_id]['scene_dict'] and prev_indent in split_meta[each_id]['dialog_dict']:
                        new_sections.append(('DIALOG', section_name, section_texts))
                        num_dialogs += 1
                        section_name = 'NULL'
                        section_texts = ''
                        new_scene = True

                    section_texts += re.sub(r'\t', '', sent).strip() + '\n'
                    prev_indent = indent
                    
                if new_scene:
                    new_sections.append(('SCENE', section_name, section_texts))
                    num_scenes += 1
                else:
                    if indent in split_meta[each_id]['dialog_dict']:
                        new_sections.append(('DIALOG', section_name, section_texts))
                        num_dialogs += 1
                    elif indent in split_meta[each_id]['scene_dict']:
                        new_sections.append(('SCENE', section_name, section_texts))
                        num_scenes += 1
            # print('[*] {}:  # of scenes = {},  # of dialogs = {}'.format(each_id, num_scenes, num_dialogs))

        # save
        total_scenes += num_scenes
        total_dialogs += num_dialogs
        if split_meta[each_id]['type'].startswith('Original'):
            filename = Path('tmp/silver/{}.txt'.format(each_id))
        elif split_meta[each_id]['type'].startswith('Reorg'):
            filename = Path('tmp/by_stats/{}.txt'.format(each_id))
        elif split_meta[each_id]['type'] == 'Fail':
            filename = Path('tmp/bad_format/{}.txt'.format(each_id))
        else:
            raise "Unrecognized type"
        with open(filename, 'w', encoding='cp1256') as fp:
            for new_section in new_sections:
                fp.write(json.dumps(new_section))
                fp.write('\n')
    print('[*] Total # of scenes = {}'.format(total_scenes))
    print('[*] Total # of dialogs = {}'.format(total_dialogs))


def create_split_pkl():
    for folder in ["bad_format", "by_stats", "silver"]:
        output_file = "./tmp/{}_imsdb_self_collected_add_space.pkl".format(folder)
        if Path(output_file).exists():
            continue
        rows_list = list()
        for script_file in glob.glob("./tmp/{}/*.txt".format(folder)):
            with open(script_file, 'r') as fp:
                book_id = Path(script_file).stem
                lines = fp.readlines()
                for sec_idx, line in enumerate(lines):
                    sample = json.loads(line)
                    rows_list.append({
                        'label':        sample[0].lower(),
                        'title':        sample[1],
                        'text':         sample[2].replace('\n', ' '),
                        'movie_name':   book_id,
                        'section_id':   sec_idx,
                        'sentence_id':  0,
                    })
        train_df = pd.DataFrame(
            rows_list,
            columns=['label', 'title', 'text', 'movie_name', 'section_id','sentence_id']
        )
        print('[*] # of samples in {}: {}'.format(folder, train_df.shape[0]))
        print('[*] Extracting entities in titles and texts, which may take 50-100 mins ... ')
        import time 
        tic = time.time()
        train_df['NER_title'] = train_df['title'].apply(spacy_entity)
        train_df['NER_text'] = train_df['text'].apply(spacy_entity)
        toc = time.time()
        print('Time eclipsed: {:.2f}'.format(toc - tic))
        train_df.to_pickle(output_file)
        print('[*] Saved in <{}>'.format(output_file))


def _clean_movie_names(raw):
        # Remove the content in brackets
        # Remove any white spaces at the end of the string
        # Convert to lowercase
        return re.sub(r"\(.*\)","", raw).strip().lower()


def _filter_movie_characters(row):
    agreement = 60
    cols = ['I', 'N', 'F', 'P', 'E', 'S', 'T', 'J']
    if any(row[c] >= agreement for c in cols):
        if row['vote_count_mbti'] >= 3:
            return True
    return False


def _name_match(name_str1, name_str2):
    match_success = False
    tokens = name_str1.split()
    char_name_tokens = name_str2.split()
    
    name1 = set([token.text for token in NLP_PIPE.tokenizer(name_str1) if any(c.isalpha() for c in token.text)])
    name2 = set([token.text for token in NLP_PIPE.tokenizer(name_str2) if any(c.isalpha() for c in token.text)])
    union = name1 & name2
    return union == name1 or union == name2


def _map_name_to_id(row):
    global movie_characters_to_id_mapping

    subcategory = row['movie_name']
    name_list = row['NER_text']
    character_ids = set()
    if subcategory not in movie_characters_to_id_mapping:
        return character_ids
    else:
        mapping = movie_characters_to_id_mapping[subcategory]
        for name in name_list:
            if name[1] != 'PERSON':
                continue
            name = name[0].lower()
            for full_name in mapping.keys():
                if _name_match(name,full_name):
                    # nick name in full name
                    # TODO:improve
                    character_ids.add((mapping[full_name], name))
        return list(character_ids)


def _title_map_name_to_id(row):
    global movie_characters_to_id_mapping

    subcategory = row['movie_name']
    name_list = row['NER_title']
    character_ids = set()
    if subcategory not in movie_characters_to_id_mapping:
        return character_ids
    else:
        mapping = movie_characters_to_id_mapping[subcategory]
        for name in name_list:
            if name[1] != 'PERSON':
                continue
            name = name[0].lower()
            for full_name in mapping.keys():
                if _name_match(name,full_name):
                    # nick name in full name
                    # TODO:improve
                    character_ids.add((mapping[full_name], name))
        return list(character_ids)

@DeprecationWarning
def load_MBTI_data_for_superhero():
    movie_characters_df = pd.read_pickle('./preprocessed/Movie_superhero.pkl')
    # clean movie names
    movie_characters_df['subcategory'] = movie_characters_df['subcategory'].apply(_clean_movie_names)
    # clean movie characters by agreements
    # if any column agreement > 60
    cleaned_movie_characters_df_60 = movie_characters_df[movie_characters_df.apply(_filter_movie_characters, axis=1)]
    MBTI_dict = cleaned_movie_characters_df_60.to_dict('records')
    return MBTI_dict


def build_mapping():
    movie_characters_df = pd.read_pickle('./preprocessed/Movie_superhero.pkl')
    movie_characters_df['subcategory'] = movie_characters_df['subcategory'].apply(_clean_movie_names)
    
    movie_characters_to_id_mapping = {}
    movie_character_id_to_name_mapping = {}
    for i in range(movie_characters_df.shape[0]):
        character = movie_characters_df['mbti_profile'].values[i]
        subcategory = movie_characters_df['subcategory'].values[i]
        _id = movie_characters_df['id'].values[i]
        if subcategory not in movie_characters_to_id_mapping:
            movie_characters_to_id_mapping[subcategory] = dict()
            movie_character_id_to_name_mapping[subcategory] = dict()
        cleaned_character = character.lower()
        movie_characters_to_id_mapping[subcategory][cleaned_character] = _id
        movie_character_id_to_name_mapping[subcategory][_id] = cleaned_character
    return movie_characters_to_id_mapping, movie_character_id_to_name_mapping


def merge_data():
    ### BookQA part ###
    scene_df = pickle.load(open('./preprocessed/bookQA_NER_add_space.pkl', "rb"))
    # import storedScript.csv
    movie_name_to_id_mapping = pd.read_csv('./narrative_qa/storedScript.csv')
    scene_df = scene_df.merge(movie_name_to_id_mapping, left_on="book_id", right_on="id", how="left")
    scene_df.drop('id', axis=1, inplace=True)
    scene_df.drop('book_id', axis=1, inplace=True)
    scene_df = scene_df.rename(columns={'movieName': 'movie_name'})
    scene_df['source'] = 'old'

    ### silver part ###
    silver_df = pd.read_pickle('./tmp/silver_imsdb_self_collected_add_space.pkl')
    silver_df = silver_df.rename(columns={'label': 'predsWithTitle'})
    silver_df.drop('section_id', axis=1, inplace=True)
    silver_df.drop('sentence_id', axis=1, inplace=True)
    silver_df['source'] = 'new'

    ### by_stat part ###
    by_stats_df = pd.read_pickle('./tmp/by_stats_imsdb_self_collected_add_space.pkl')
    by_stats_df = by_stats_df.rename(columns={'label': 'predsWithTitle'})
    by_stats_df.drop('section_id', axis=1, inplace=True)
    by_stats_df.drop('sentence_id', axis=1, inplace=True)
    by_stats_df['source'] = 'new'
    
    ### bad_format part ###
    bad_format_df = pd.read_pickle('./tmp/bad_format_imsdb_self_collected_add_space.pkl')
    bad_format_df = bad_format_df.rename(columns={'label': 'predsWithTitle'})
    bad_format_df.drop('section_id', axis=1, inplace=True)
    bad_format_df.drop('sentence_id', axis=1, inplace=True)
    bad_format_df['source'] = 'new'

    ### Merge ###
    merged_df = pd.concat(
        [scene_df, silver_df, by_stats_df, bad_format_df], 
        ignore_index = True
    )
    merged_df.movie_name[merged_df.movie_name=='star wars the empire strikes back'] = 'star wars'
    merged_df.movie_name[merged_df.movie_name=='star wars return of the jedi'] = 'star wars'
    merged_df.movie_name[merged_df.movie_name=='star wars the phantom menace'] = 'star wars'
    merged_df.movie_name[merged_df.movie_name=='star wars attack of the clones'] = 'star wars'
    merged_df.movie_name[merged_df.movie_name=='star wars the force awakens'] = 'star wars'
    merged_df.movie_name[merged_df.movie_name=='star wars a new hope'] = 'star wars'
    merged_df.movie_name[merged_df.movie_name=='star trek ii the wrath of khan'] = 'star trek'
    merged_df.movie_name[merged_df.movie_name=='star trek the motion picture'] = 'star trek'
    merged_df.movie_name[merged_df.movie_name=='star trek nemesis'] = 'star trek'
    merged_df.movie_name[merged_df.movie_name=='star trek first contact'] = 'star trek'
    merged_df.movie_name[merged_df.movie_name=='star trek generations'] = 'star trek'
    merged_df.movie_name[merged_df.movie_name=='lord of the rings return of the king'] = 'the lord of the rings'
    merged_df.movie_name[merged_df.movie_name=='lord of the rings the two towers'] = 'the lord of the rings'
    merged_df.movie_name[merged_df.movie_name=='the lord of the rings fellowship of the ring'] = 'the lord of the rings'
    merged_df.movie_name[merged_df.movie_name=='austin powers international man of mystery'] = 'austin powers'
    merged_df.movie_name[merged_df.movie_name=='austin powers the spy who shagged me'] = 'austin powers'
    merged_df.movie_name[merged_df.movie_name=='pirates of the caribbean dead mans chest'] = 'pirates of the caribbean'
    merged_df.movie_name[merged_df.movie_name=='hellraiser 3 hell on earth'] = 'hellraiser'
    merged_df.movie_name[merged_df.movie_name=='the x files fight the future'] = 'x files'
    merged_df.movie_name[merged_df.movie_name=='guardians of the galaxy vol 2'] = 'mcu: the heroes'
    merged_df.movie_name[merged_df.movie_name=='how to train your dragon 2'] = 'how to train your dragon'
    merged_df.movie_name[merged_df.movie_name=='indiana jones and the raiders of the lost ark'] = 'indiana jones'
    merged_df.movie_name[merged_df.movie_name=='indiana jones and the temple of doom'] = 'indiana jones'
    merged_df.movie_name[merged_df.movie_name=='indiana jones iv'] = 'indiana jones'
    merged_df.movie_name[merged_df.movie_name=='airplane 2 the sequel'] = 'airplane!'
    merged_df.movie_name[merged_df.movie_name=='godfather part ii'] = 'the godfather'
    merged_df.movie_name[merged_df.movie_name=='mission impossible ii'] = 'mission: impossible'
    merged_df.movie_name[merged_df.movie_name=='the godfather part iii'] = 'the godfather'
    merged_df.movie_name[merged_df.movie_name=='final destination 2'] = 'final destination'
    merged_df.movie_name[merged_df.movie_name=='boondock saints 2 all saints day'] = 'the boondock saints'
    merged_df.movie_name[merged_df.movie_name=='scream 2'] = 'scream'
    merged_df.movie_name[merged_df.movie_name=='scream 3'] = 'scream'
    merged_df.movie_name[merged_df.movie_name=='despicable me 2'] = 'despicable me'
    merged_df.movie_name[merged_df.movie_name=='friday the 13th part viii jason takes manhattan'] = 'friday the 13th'
    merged_df.movie_name[merged_df.movie_name=='terminator 2 judgement day'] = 'terminator'
    merged_df.movie_name[merged_df.movie_name=='terminator salvation'] = 'terminator'
    merged_df.movie_name[merged_df.movie_name=='die hard 2'] = 'die hard'
    merged_df.movie_name[merged_df.movie_name=='hellboy 2 the golden army'] = 'hellboy'
    merged_df.movie_name[merged_df.movie_name=='american shaolin king of kickboxers ii'] = 'american shaolin'
    merged_df.movie_name[merged_df.movie_name=='mad max 2 the road warrior'] = 'mad max'
    merged_df.movie_name[merged_df.movie_name=='jurassic park the lost world'] = 'jurassic park / jurassic world'
    merged_df.movie_name[merged_df.movie_name=='jurassic park'] = 'jurassic park / jurassic world'
    merged_df.movie_name[merged_df.movie_name=='transformers the movie'] = 'transformers'
    merged_df.movie_name[merged_df.movie_name=='cars 2'] = 'cars'
    merged_df.movie_name[merged_df.movie_name=='twilight new moon'] = 'twilight'
    merged_df.movie_name[merged_df.movie_name=='alien resurrection'] = 'alien'
    merged_df.movie_name[merged_df.movie_name=='robin hood prince of thieves'] = 'robin hood'
    merged_df.movie_name[merged_df.movie_name=='chronicles of narnia the lion the witch and the wardrobe'] = 'the chronicles of narnia'
    merged_df.movie_name[merged_df.movie_name=='shrek the third'] = 'shrek'
    merged_df.movie_name[merged_df.movie_name=='gi joe the rise of cobra'] = 'g.i. joe'
    merged_df.movie_name[merged_df.movie_name=='tron legacy'] = 'tron'
    merged_df.movie_name[merged_df.movie_name=='ghostbusters 2'] = 'ghostbusters'
    merged_df.movie_name[merged_df.movie_name=='saw'] = 'saw series'
    merged_df.movie_name[merged_df.movie_name=='men in black 3'] = 'men in black'
    merged_df.movie_name[merged_df.movie_name=='the world is not enough'] = 'james bond'
    merged_df.movie_name[merged_df.movie_name=='tomorrow never dies'] = 'james bond'
    merged_df.movie_name[merged_df.movie_name=='batman 2'] = 'dc extended universe'
    merged_df.movie_name[merged_df.movie_name=='batman'] = 'dc extended universe'
    merged_df.movie_name[merged_df.movie_name=='wonder woman'] = 'dc extended universe'
    merged_df.movie_name[merged_df.movie_name=='spider man'] = 'mcu: the heroes'
    merged_df.movie_name[merged_df.movie_name=='the avengers'] = 'mcu: the heroes'
    merged_df.movie_name[merged_df.movie_name=='the avengers(2012)'] = 'mcu: the heroes'
    merged_df.movie_name[merged_df.movie_name=='black panther'] = 'mcu: the heroes'
    merged_df.movie_name[merged_df.movie_name=='thor'] = 'mcu: the heroes'
    merged_df.movie_name[merged_df.movie_name=='thor ragnarok'] = 'mcu: the heroes'
    merged_df.movie_name[merged_df.movie_name=='alien vs predator'] = 'alien vs. predator'
    merged_df.movie_name[merged_df.movie_name=='the matrix reloaded'] = 'the matrix trilogy'
    merged_df.movie_name[merged_df.movie_name=='the matrix'] = 'the matrix trilogy'
    merged_df.movie_name[merged_df.movie_name=='godzilla'] = 'monsterverse'
    merged_df.movie_name[merged_df.movie_name=='aliens'] = 'alien'
    merged_df.movie_name[merged_df.movie_name=='blade trinity'] = 'blade'
    merged_df.movie_name[merged_df.movie_name=='box the'] = 'the box'
    merged_df.movie_name[merged_df.movie_name=='the crow city of angels'] = 'the crow: city of angels'
    merged_df.movie_name[merged_df.movie_name=='i robot'] = 'i, robot'
    merged_df.movie_name[merged_df.movie_name=='la confidential'] = 'l.a. confidential'
    merged_df.movie_name[merged_df.movie_name=='sweeney todd the demon barber of fleet street'] = 'sweeney todd: the demon barber of fleet street'
    merged_df.movie_name[merged_df.movie_name=='the talented mr ripley'] = 'the talented mr. ripley'
    merged_df.movie_name[merged_df.movie_name=='fright night (2011 film)'] = 'fright night'

    return merged_df


def add_character_ids(df):
    print('[*] Using character names and moive names to identify unique character ids, which may take ~15 mins ...')
    df['title_character_ids'] = df.apply(_title_map_name_to_id, axis=1)
    df['character_ids'] = df.apply(_map_name_to_id, axis=1)
    return df


def generate_dialog_dict(all_data_df):
    dialog_dict = {}
    for i in range(all_data_df.shape[0]):
        title = all_data_df['title'].values[i].strip('\"').strip().lower()
        text = all_data_df['text'].values[i].strip('\"').strip()
        movie_name = all_data_df['movie_name'].values[i]
        sec_type = all_data_df['predsWithTitle'].values[i]
        if title in ('null', '\"\"', ''):
            continue
        if movie_name not in movie_characters_to_id_mapping:
            continue
        if sec_type != 'dialog':
            continue
        if movie_name not in dialog_dict:
            dialog_dict[movie_name] = {}
            
        for full_name in movie_characters_to_id_mapping[movie_name].keys():
            if _name_match(title,full_name):
                if full_name not in dialog_dict[movie_name]:
                    dialog_dict[movie_name][full_name] = {text}
                else:
                    dialog_dict[movie_name][full_name].add(text)
    return dialog_dict


def generate_scene_dict(all_data_df):
    scene_dict = {}
    for i in range(all_data_df.shape[0]):
        sec_type = all_data_df['predsWithTitle'].values[i]
        text = all_data_df['text'].values[i].strip('\"').strip()
        subcategory = all_data_df['movie_name'].values[i]
        character_ids = all_data_df['character_ids'].values[i]

        if subcategory not in movie_characters_to_id_mapping:
            continue
        if sec_type != 'scene':
            continue
        if subcategory not in scene_dict:
            scene_dict[subcategory] = {}

        for (character_id,char_mention) in character_ids:
            character_name = movie_character_id_to_name_mapping[subcategory][character_id]
            if character_name not in scene_dict[subcategory]:
                scene_dict[subcategory][character_name] = {(char_mention,text)}
            else:
                scene_dict[subcategory][character_name].add((char_mention,text))
    return scene_dict


def generate_mention_dict(all_data_df):
    mention_dict = {}
    for i in range(all_data_df.shape[0]):
        sec_type = all_data_df['predsWithTitle'].values[i]
        text = all_data_df['text'].values[i].strip('\"').strip()
        subcategory = all_data_df['movie_name'].values[i]
        character_ids = all_data_df['character_ids'].values[i]

        if subcategory not in movie_characters_to_id_mapping:
            continue
        if sec_type != 'dialog':
            continue
        if subcategory not in mention_dict:
            mention_dict[subcategory] = {}

        for (character_id,char_mention) in character_ids:
            character_name = movie_character_id_to_name_mapping[subcategory][character_id]
            if character_name not in mention_dict[subcategory]:
                mention_dict[subcategory][character_name] = {(char_mention,text)}#set
            else:
                mention_dict[subcategory][character_name].add((char_mention,text))
    return mention_dict


def save(dialog_dict, scene_dict, mention_dict):
    with open('dialog_dict.pickle', 'wb') as handle:
        pickle.dump(dialog_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('[*] Save data in <dialog_dict.pickle>')
    with open('scene_dict.pickle', 'wb') as handle:
        pickle.dump(scene_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('[*] Save data in <scene_dict.pickle>')
    with open('mention_dict.pickle', 'wb') as handle:
        pickle.dump(mention_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('[*] Save data in <mention_dict.pickle>')
    



if __name__ == '__main__':

    SPLIT_SIGN = analyze_segmentation_signals()
    read_and_split_documents(SPLIT_SIGN)
    create_split_pkl()

    (
        movie_characters_to_id_mapping,
        movie_character_id_to_name_mapping,
    ) = build_mapping()
    merged_df = merge_data()
    merged_df = add_character_ids(merged_df)

    dialog_dict = generate_dialog_dict(merged_df)
    scene_dict = generate_scene_dict(merged_df)
    mention_dict = generate_mention_dict(merged_df)

    save(dialog_dict, scene_dict, mention_dict)