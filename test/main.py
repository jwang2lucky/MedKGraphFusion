import os
import json
import ijson
import logging
from argparse import ArgumentParser

from model import KnowledgeGraphLLM
from step_01 import step_01_concept_extraction
from step_02 import build_key_offset_index,step_02_triple_extraction
from step_03 import step_03_fusion


def stream_concept_abstracts(file_path):
    """
    流式读取超大 JSON（63GB），每次 yield 一个 {concept_name: info}
    自动跳过损坏的 JSON 项，避免 IncompleteJSONError
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            parser = ijson.kvitems(f, '')  # 只读最外层 key-value
            for concept, info in parser:
                yield {concept: info}

        except ijson.IncompleteJSONError as e:
            print("⚠ 跳过损坏的 JSON 项:", e)
            return

class LazyJSONDict:
    def __init__(self, json_path, index_path):
        self.json_path = json_path
        self.index = json.load(open(index_path))

    def get(self, concept):
        """使用 offset 快速定位 concept JSON 对象"""
        if concept not in self.index:
            return None

        offset = self.index[concept]

        with open(self.json_path, "rb") as f:
            f.seek(offset)
            # 从偏移位置读取 JSON 对象字符串
            chunk = b""
            brace_count = 0
            in_object = False
            while True:
                c = f.read(1)
                if not c:
                    break
                if c == b'{':
                    brace_count += 1
                    in_object = True
                elif c == b'}':
                    brace_count -= 1
                chunk += c
                if in_object and brace_count == 0:
                    break
            try:
                obj = json.loads(chunk.decode('utf-8'))
                return obj
            except Exception:
                return None
'''class LazyJSONDict:
    def __init__(self, file_path):
        self.file_path = file_path

    def get(self, concept):
        """逐条扫描 JSON，仅在匹配 key 时返回数据"""
        for item in stream_concept_abstracts(self.file_path):
            (key, val), = item.items()
            if key == concept:
                return val
        return None
'''
if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--run_name", type=str, default="test",
                          help="Assign a name to this run. The name will be used to, e.g., determine "
                               "the output directory. We recommend to use unique and descriptive names "
                               "to distinguish the results of different models.")
    argparse.add_argument("--dataset", type=str,
                          required=True,
                          help="Name of the dataset. Is used to, e.g., determine the input directory.")
    argparse.add_argument("--relation_definitions_file", type=str,
                          required=True,
                          help="Path to the relation definitions file. The file should be a JSON file, "
                               "where the keys are the relation types and the values are dictionaries "
                               "with the following keys: 'label', 'description'.")

    # these arguments allow to provide the data from the previous steps directly
    # instead of running these steps
    argparse.add_argument("--input_json_file", type=str, default="",
                          help="Path to the input file. Step 1 will be skipped if this argument "
                               "is provided. The input file should be a JSON file with the "
                               "following structure: "
                               "{'concept1': [{'abstract': ['abstract1', ...], 'label: 0},...} "
                               "E.g. data/test/concept_abstracts.json is the associated file created"
                               "during step 1 in the test run.")
    argparse.add_argument("--input_triple_file", type=str, default="",
                          help="Path to the input file storing the triples in the format as outputted "
                               "by the candidate triple extraction model. Step 1 and step 2 will "
                               "be skipped if this argument is provided.")

    # these arguments allow to configure the LLM model
    argparse.add_argument("--model", type=str, default="/home/wangjie/KGC/Graphusion-main/models/BioGPT",
                          help="Name of the LLM that should be used for the KG construction.")
    argparse.add_argument("--max_resp_tok", type=int, default=200,
                          help="Maximum number of tokens in the response of the candidate triple "
                               "extraction model.")
    argparse.add_argument("--max_input_char", type=int, default=10000,
                          help="Maximum number of characters in the input of the candidate triple "
                               "extraction model.")
    argparse.add_argument("--prompt_tpextraction", type=str,
                          default="prompts_n/prompt_tpextraction.txt",
                          help="Path to the prompt template for step 1.")
    argparse.add_argument("--prompt_fusion", type=str, default="prompts_n/prompt_fusion.txt",
                          help="Path to the prompt template for fusion.")

    # these arguments allow to provide additional data
    argparse.add_argument("--gold_concept_file", type=str,
                          default="",
                          help="Path to a file with concepts that are provided by experts. "
                               "The file should be a tsv file, each row should look like: "
                               "'concept id | concept")
    argparse.add_argument('--refined_concepts_file', type=str,
                          default=None,
                          help='In step 2 (candidate triple extraction) many new concepts might be '
                               'added. Instead of using these, concepts can be provided through this '
                               'parameter. Specify the path to a file with refined concepts '
                               'of the graph. The file should be a tsv file, each row should look like: '
                               '"concept id | concept name"')
    argparse.add_argument("--annotated_graph_file", type=str,
                          default="data/prerequisite_of_graph.tsv",
                          help="Path to the annotated graph.")

    # language settings
    argparse.add_argument('--language', type=str, default='english',
                          help='Language of the abstracts.')

    # logging
    argparse.add_argument('--verbose', action='store_true',
                          help='Print additional information to the console.')

    # Parse the arguments
    args = argparse.parse_args()
    VERBOSE = args.verbose
    RUN_NAME = args.run_name
    RELATION_DEFINITIONS_FILE = args.relation_definitions_file
    MODEL_NAME = args.model
    MAX_RESPONSE_TOKEN_LENGTH_CANDIDATE_TRIPLE_EXTRACTION = args.max_resp_tok
    PROMPT_TPEXTRACTION_FILE = args.prompt_tpextraction
    PROMPT_FUSION_FILE = args.prompt_fusion

    # --- Setup ---
    # initialize logger
    if VERBOSE:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info(f"RUN_NAME: {RUN_NAME}")

    # Prepare the output directory
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs-01')
    if not os.path.exists(f'./outputs/{RUN_NAME}'):
        os.makedirs(f'./outputs/{RUN_NAME}')

    # write config to output directory
    config = args.__dict__
    json.dump(config, open(f'./outputs/{RUN_NAME}/config.json', 'w'), indent=4)

    # assign output file names if not provided
    
    if args.input_json_file == "":
        CONCEPT_EXTRACTION_OUTPUT_FILE = f'./outputs/{RUN_NAME}/concepts.tsv'
        CONCEPT_ABSTRACTS_OUTPUT_FILE = f'./outputs/{RUN_NAME}/concept_abstracts.json'
        CONCEPT_INDEX_OUTPUT_FILE = f'./outputs/{RUN_NAME}/concept_abstracts.index'   #添加index文件目录
        build_key_offset_index(CONCEPT_ABSTRACTS_OUTPUT_FILE,CONCEPT_INDEX_OUTPUT_FILE)
    else:
        CONCEPT_ABSTRACTS_OUTPUT_FILE = args.input_json_file
        logging.info(
            f"Using provided input file: {CONCEPT_ABSTRACTS_OUTPUT_FILE} "
            f"(skipping step 1 - concept extraction).")

    if args.input_triple_file == "":
        TRIPLE_EXTRACTION_OUTPUT_FILE = f'./outputs/{RUN_NAME}/step-02.jsonl'
    else:
        TRIPLE_EXTRACTION_OUTPUT_FILE = args.input_triple_file
        logging.info(
            f"Using provided input file: {TRIPLE_EXTRACTION_OUTPUT_FILE} "
            f"(skipping step 2 - triple extraction).")

    # output file of the pipeline
    FUSION_OUTPUT_FILE = f'./outputs/{RUN_NAME}/step-03.jsonl'

    # Load the relation definitions
    relation_def = json.load(open(RELATION_DEFINITIONS_FILE, 'r'))
    relation_types = list(relation_def.keys())
    relation_2_id = {v: k for k, v in enumerate(relation_types)}
    id_2_relation = {k: v for k, v in enumerate(relation_types)}

    # Configure API keys
    #os.environ["CEREBRAS_API_KEY"] = json.load(open('private_config.json'))['CEREBRAS_API_KEY']

    # init the LLM
    model = KnowledgeGraphLLM(model_name=MODEL_NAME,
                              max_tokens=MAX_RESPONSE_TOKEN_LENGTH_CANDIDATE_TRIPLE_EXTRACTION)

    # --- Pipeline ---
    # 配置你的路径
    BASE_DIR = "/home/MedKGFusion"
    INPUT_DIR = os.path.join(BASE_DIR, "../text_data/text_data")
    OUTPUT_DIR = "./outputs"
    #INPUT_DIR = os.path.join(BASE_DIR, "Evaluation_MedGraphusion/data")
    #OUTPUT_DIR = "./outputs-N"
    # 可以在这里修改具体的本体和模型路径
    CONFIG_01 = {
        "ontology_file": os.path.join(BASE_DIR, "../text_data/ontology_data/extracted_str_en_unique.txt"),
        "embedding_path": os.path.join(BASE_DIR, "SapBERT_local_test"), 
        "gold_concept_file": "../text_data/database_entities.tsv",
        "stop_words": "english"
    }

    if not os.path.exists(INPUT_DIR):
        logger.error(f"Input directory not found: {INPUT_DIR}")
        exit(1)

    for file_name in os.listdir(INPUT_DIR):
        if not file_name.endswith(".txt"):
            continue
            
        logger.info(f"Processing file: {file_name}")
        input_file = os.path.join(INPUT_DIR, file_name)
        
        sample_texts = []
        origins = []

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    parts = line.strip().split("|||")
                    
                    # 鲁棒的解析逻辑
                    if len(parts) >= 3:
                        o_id, title, abstract = parts[0], parts[1], parts[2]
                    elif len(parts) == 2:
                        o_id, abstract = parts[0], parts[1]
                    else:
                        continue
                        
                    sample_texts.append(abstract)
                    origins.append(o_id)
        except Exception as e:
            logger.error(f"Error reading file {input_file}: {e}")
            continue
    # extract concepts
    step_01_concept_extraction(texts=sample_texts,
                               concept_extraction_output_file=CONCEPT_EXTRACTION_OUTPUT_FILE,
                               concept_abstracts_output_file=CONCEPT_ABSTRACTS_OUTPUT_FILE,
                               logging=logging,
                               origins=origins, 
                               config=CONFIG_01)

    data_generator = stream_concept_abstracts(CONCEPT_ABSTRACTS_OUTPUT_FILE)

    # step2 调用时可以按 concept 批量传递
    # 假设每次 step2 想处理 1000 个 concept
    batch_size = 1000
    batch = {}
    for concept_dict in data_generator:
        batch.update(concept_dict)
        if len(batch) >= batch_size:
            step_02_triple_extraction(model=model,
                                      output_file=TRIPLE_EXTRACTION_OUTPUT_FILE,
                                      relation_def=relation_def,
                                      data=batch, 
                                      logging=logging,
                                      config=config)
            batch = {}

    # 剩余不满 batch_size 的概念
    if batch:
        step_02_triple_extraction(model=model,
                                  output_file=TRIPLE_EXTRACTION_OUTPUT_FILE,
                                  relation_def=relation_def,
                                  data=batch,
                                  logging=logging,
                                  config=config)
    
    # --- Step3: Fusion ---
    data_for_fusion = LazyJSONDict(CONCEPT_ABSTRACTS_OUTPUT_FILE,CONCEPT_INDEX_OUTPUT_FILE)

    step_03_fusion(
        model=model,
        input_file=TRIPLE_EXTRACTION_OUTPUT_FILE,
        output_file=FUSION_OUTPUT_FILE,
        relation_def=relation_def,
        relation_2_id=relation_2_id,
        data=data_for_fusion,   # ❤️ 现在不会 OOM,根据索引减少读取耗时
        logging=logging,
        config=config)
    