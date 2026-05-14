import json
from langchain_core.prompts import ChatPromptTemplate
from collections import Counter

from tqdm import tqdm

import re
import os


def build_key_offset_index(json_path, index_path,
                           chunk_size=1024*1024,
                           overlap=200):

    """
    为超大 JSON 构建 key→offset 索引
    """

    filesize = os.path.getsize(json_path)
    print(f"开始构建索引... 文件大小: {filesize/1024/1024:.2f} MB")

    index = {}
    key_pattern = re.compile(r'"([^"]+)"\s*:\s*\{')

    with open(json_path, "rb") as f:

        pos = 0
        pbar = tqdm(total=filesize, unit='B', unit_scale=True)

        while pos < filesize:
            f.seek(pos)
            chunk = f.read(chunk_size + overlap).decode('utf-8', errors='ignore')

            for m in key_pattern.finditer(chunk):
                concept = m.group(1)
                key_offset = pos + m.start()
                index[concept] = key_offset

                # 每 2000 个打印一次防止“以为没运行”
                if len(index) % 2000 == 0:
                    print(f"已扫描 {len(index)} 个 key...")

            pos += chunk_size
            pbar.update(chunk_size)

        pbar.close()

    print(f"扫描完成，共找到 {len(index)} 个概念，正在写出索引...")

    with open(index_path, "w") as f:
        json.dump(index, f)

    print(f"索引已保存到: {index_path}")

def step_02_triple_extraction(model: any,
                              output_file: str,
                              relation_def: dict[str, dict[str, str]],
                              data: dict[str, dict[str, list[str]]],
                              logging: any,
                              config: dict[str, any]):
    """
    Step 1: Candidate Triple Extraction
    Extracts candidate triples from the data and writes them to the output file.

    :param model: the language model to use
    :param output_file: the file to write the extracted triples to
    :param relation_def: the relation definitions
    :param data: the data to extract triples from
    :param logging: the logger
    :param config: the configuration can be provided with the following keys: prompt_tpextraction,
    max_input_char

    :return: None
    """

    if 'prompt_tpextraction' not in config:
        config['prompt_tpextraction'] = "/mnt/gpu04_data/wangjie/KGC/Graphusion-main/prompts/prompt_tpextraction.txt"
        logging.info(f"No prompt template for triple extraction provided. "
                     f"Using default prompt: {config['prompt_tpextraction']}")
    if 'max_input_char' not in config:
        config['max_input_char'] = 10000
        logging.info(f"No max_input_char provided. Using default value: {config['max_input_char']}")


    logging.info("Step 1: Starting candidate triple extraction.")
    output_stream = open(output_file, 'w')
    #debug_stream = open("debug_model_outputs.txt", 'a', encoding='utf-8')

    # initialize the prompt template
    prompt_template_txt = open(config['prompt_tpextraction']).read()
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledge graph builder."),
        ("user", prompt_template_txt)
    ])

    # iterate over the data, extract triples and write them to the output stream
    extracted_relations = []
    for concept_id, (concept_name, concept_data) in tqdm(enumerate(data.items()), total=len(data)):
        abstracts = ' '.join(data[concept_name]['abstracts'])

        # instantiate the prompt template

        prompt = prompt_template.invoke(
            {"abstracts": abstracts[:config['max_input_char']],
             "concepts": [concept_name],
             "relation_definitions": '\n'.join(
                 [f"{rel_type}: {rel_data['description']}" for rel_type, rel_data in
                  relation_def.items()])})
        # query the model
        response = model.invoke(prompt)
        # debug输出
        #debug_stream.write(f"Concept: {concept_name}\nPrompt:\n{prompt}\nResponse:\n{response}\n{'='*80}\n")
        #debug_stream.flush()

        if response != "None":
            try:
                response_json = json.loads(response)
            except json.JSONDecodeError as e:
                logging.warning(f"JSON decode failed for concept {concept_name}: {e}")
                response_json = []
            for triple in response_json:
                if not isinstance(triple, dict) or 'p' not in triple or 's' not in triple or 'o' not in triple:
                    logging.warning(f"Invalid or empty triple skipped: {triple}")
                    continue

                if triple['p'] not in list(relation_def.keys()):
                    continue
                else:
                    extracted_relations.append(triple['p'])

                triple['id'] = concept_id
                triple['concept'] = concept_name
                triple['origins'] = concept_data.get('origins', [])
                output_stream.write(json.dumps(triple) + '\n')
                print(f"[TRIPLE] {triple}")
    output_stream.close()
    #debug_stream.close()

    logging.info("Step 1: Candidate Triple Extraction completed.")
    logging.info(f"Num extracted candidate triples: {len(extracted_relations)}")
    #logging.debug(f"Extracted candidate triples by relaton type: {Counter(extracted_relations)}")
if __name__ == "__main__":
