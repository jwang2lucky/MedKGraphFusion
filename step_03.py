import json
import pandas as pd
import random
import os
from collections import defaultdict
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
import time
# 引入 graphs 中的函数 (不再使用 verbalize_neighbors_triples_from_triples)
from graphs import get_nx_graph, verbalize_neighbors_triples_from_graph
import re

def extract_json_array(text):
    """
    强力提取 JSON 数组。
    即使 LLM 输出被截断（缺少结尾的 ]），也能抢救出前面已经生成的完整对象。
    """
    text = text.strip()
    
    # ---------------------------------------------------------
    # 策略 1: 尝试直接解析 (最完美的情况)
    # ---------------------------------------------------------
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ---------------------------------------------------------
    # 策略 2: 尝试提取 Markdown 代码块 (```json ... ```)
    # ---------------------------------------------------------
    if "```" in text:
        pattern = r"```(?:json)?\s*(\[.*?\])\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

    # ---------------------------------------------------------
    # 策略 3: 流式解析 (Stream Rescue Mode) - 专治截断数据
    # ---------------------------------------------------------
    # 原理：找到第一个 [，然后用 decoder 一个个解析里面的 {}
    # ---------------------------------------------------------
    try:
        start_idx = text.find('[')
        if start_idx == -1:
            return None # 连个开头都没有，没法救

        decoder = json.JSONDecoder()
        # 从 [ 后面开始扫描
        scan_idx = start_idx + 1
        collected_objects = []

        while scan_idx < len(text):
            # 1. 跳过空白字符
            while scan_idx < len(text) and text[scan_idx].isspace():
                scan_idx += 1
            
            if scan_idx >= len(text):
                break # 读到头了

            # 2. 如果遇到逗号，跳过
            if text[scan_idx] == ',':
                scan_idx += 1
                continue
            
            # 3. 如果遇到结尾 ]，说明是完整的，直接结束
            if text[scan_idx] == ']':
                break

            # 4. 尝试解析下一个 JSON 对象
            try:
                # raw_decode 会从 scan_idx 开始解析一个合法的 JSON 对象
                # 并返回 (对象, 解析结束后的新位置)
                obj, end_idx = decoder.raw_decode(text, scan_idx)
                collected_objects.append(obj)
                scan_idx = end_idx # 移动指针到对象后面
            except json.JSONDecodeError:
                # ★ 关键：如果这里报错，说明当前这个对象被截断了（烂尾了）
                # 我们直接停止，返回之前已经成功挽救的对象
                # logging.info("Detected truncated JSON, stopping rescue.")
                break
        
        # 只要救回来至少一个对象，就算成功
        if len(collected_objects) > 0:
            return collected_objects

    except Exception as e:
        # print(f"Rescue failed: {e}")
        pass
        
    return None
def step_03_fusion(model: any,
                   input_file: str,
                   output_file: str,
                   relation_def: dict[str, dict[str, str]],
                   relation_2_id: dict[str, int],
                   data: dict[str, dict[str, list[str]]],
                   logging: any,
                   config: dict[str, any]):

    logging.info("Step 3: Fusion started (Optimized Indexing Version).")
    
    # --- 配置默认值 ---
    config.setdefault('refined_concepts_file', None)
    config.setdefault('annotated_graph_file', "")
    config.setdefault('prompt_fusion', "prompts_n/prompt_fusion.txt")
    config.setdefault('max_input_char', 3000)

    # ---------------------------------------------------------------------
    # 1. 加载数据并构建索引 (解决慢的核心)
    # ---------------------------------------------------------------------
    logging.info("Loading triples and building index...")
    
    candidate_triples = []
    triple2origins = {}
    
    # ★ 关键优化：建立倒排索引 { concept: ["(s,p,o)", "(s,p,o)"] }
    # 这样就不需要遍历几十万行去找邻居了
    concept_to_triples_str = defaultdict(list)
    TRIPLE_VERB_TEMPLATE = "({head},{relation},{tail})\n"

    with open(input_file, 'r') as f:
        for line in f:
            t = json.loads(line)
            s, p = t['s'], t['p']
            o_val = t['o']
            # 处理列表形式的 object
            if isinstance(o_val, list):
                o_val = ", ".join(str(x) for x in o_val)
            
            # 记录三元组唯一键和来源 (你需要的 origin 逻辑)
            key = (s, p, o_val)
            candidate_triples.append(key)
            if key not in triple2origins:
                triple2origins[key] = set()
            triple2origins[key].update(t.get("origins", []))

            # ★ 构建索引: 预先格式化好字符串
            formatted_triple = TRIPLE_VERB_TEMPLATE.format(head=s, relation=p, tail=o_val)
            concept_to_triples_str[s].append(formatted_triple)
            concept_to_triples_str[o_val].append(formatted_triple)

    # ---------------------------------------------------------------------
    # 2. 确定要处理的概念
    # ---------------------------------------------------------------------
    if config['refined_concepts_file'] is not None:
        logging.info(f"Loading concepts from {config['refined_concepts_file']}")
        df = pd.read_csv(config['refined_concepts_file'], sep='|', header=None,
                         names=['id', 'concept'], index_col=0)
        id_2_concept = {i: str(c['concept']) for i, c in df.iterrows()}
    else:
        # 如果没有指定文件，使用所有涉及到的概念
        concepts = list(concept_to_triples_str.keys())
        #random.shuffle(concepts)
        #concepts = concepts[:100] # ⚠️ 测试用，正式跑请去掉 [:100]
        logging.info(f"Randomly selected {len(concepts)} concepts for processing.")
        id_2_concept = {i: c for i, c in enumerate(concepts)}

    concept_2_id = {v: k for k, v in id_2_concept.items()}

    # ---------------------------------------------------------------------
    # 3. 构建先修图 NetworkX 对象
    # ---------------------------------------------------------------------
    prerequisite_of_triples = []
    if os.path.exists(config['annotated_graph_file']):
        logging.info("Loading annotated graph...")
        with open(config['annotated_graph_file'], 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    prerequisite_of_triples.append((str(parts[0]), str(parts[1]), str(parts[2])))
    
    prerequisite_of_graph = get_nx_graph(prerequisite_of_triples, concept_2_id, relation_2_id)

    # ---------------------------------------------------------------------
    # 4. 准备 Prompt
    # ---------------------------------------------------------------------
    prompt_txt = open(config['prompt_fusion']).read()
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledge graph builder."),
        ("user", prompt_txt)
    ])
    
    relation_def_str = '\n'.join(
        [f"{rel_type}: {rel_data['description']}" for rel_type, rel_data in relation_def.items()]
    )

    # ---------------------------------------------------------------------
    # 5. 主循环
    # ---------------------------------------------------------------------
    fused_results = {}
    
    logging.info("Starting processing...")
    MAX_RETRIES = 3
    for c_id, concept in tqdm(id_2_concept.items(), total=len(id_2_concept)):
        try:
            # A. 获取候选子图 (直接查字典，速度 O(1))
            # ★ 这里的速度比 verbalize_neighbors_triples_from_triples 快几万倍
            neighbors_list = concept_to_triples_str.get(concept, [])
            candidate_subgraph = "".join(set(neighbors_list)) if neighbors_list else "None"

            # B. 获取先修图子结构
            prerequisite_subgraph_str = verbalize_neighbors_triples_from_graph(
                prerequisite_of_graph, concept, concept_2_id, id_2_concept, mode='outgoing')

            # C. 获取摘要 (使用你的 LazyJSONDict，保留了 Abstract)
            info = data.get(concept)
            abstracts = ' '.join(info['abstracts'])[:config['max_input_char']] if info else ''

            # D. 构建 Prompt 并调用模型
            prompt = prompt_template.invoke({
                "concept": concept,
                "graph1": candidate_subgraph,
                "graph2": prerequisite_subgraph_str,
                "background": abstracts,
                "relation_definitions": relation_def_str
            })
            
            response_content = None
            for attempt in range(MAX_RETRIES):
                try:
                    # 打印当前正在处理谁，方便卡死时定位
                    # print(f"Processing: {concept} (Attempt {attempt+1})") 

                    # 调用模型 (尝试设置 timeout，取决于 langchain 版本是否支持，不支持也没关系，靠下面的 except)
                    response = model.invoke(prompt) 
                    
                    response_content = response.content if hasattr(response, 'content') else str(response)
                    break # 成功了就跳出重试循环
                except Exception as e:
                    logging.warning(f"Ollama timeout/error on '{concept}', retrying ({attempt+1}/{MAX_RETRIES})... Error: {e}")
                    time.sleep(2) # 出错后歇 2 秒
            # 调用模型 (这是唯一的耗时点)
            #response = model.invoke(prompt)

            # E. 解析结果
            #content = response.content if hasattr(response, 'content') else str(response)
            if not response_content:
                logging.error(f"Skipping concept '{concept}' after {MAX_RETRIES} failures.")
                continue
            # ★★★ 增加喘息时间：防止 GPU 过热或 Ollama 队列堵死 ★★★
            time.sleep(0.5)  # 每次请求后休息 0.5 秒

            # ... (解析 JSON 和写入结果的代码 F 不变) ...
            content = response_content
            response_json = extract_json_array(response_content)
            
            if response_json is None:
                # 记录一下到底模型输出了什么导致解析失败，方便调试
                logging.warning(f"⚠️ Failed to extract JSON from concept '{concept}'. Raw output snippet: {response_content[:100]}...")
                continue
                
            # 确保解析出来的是列表，不是字典或其他
            if not isinstance(response_json, list):
                logging.warning(f"⚠️ Expected list, got {type(response_json)} for concept '{concept}'.")
                continue


            #response_json = json.loads(content)

            # F. 保存结果
            for item in response_json:
                if not isinstance(item, dict) or 'p' not in item: continue
                if item['p'] not in relation_2_id: continue

                s, p, o = item['s'], item['p'], item['o']
                unique_key = (s, p, o)

                if unique_key not in fused_results:
                    fused_results[unique_key] = {
                        "s": s, "p": p, "o": o,
                        "origins": set(),
                        "support": 0
                    }
                
                # ★ 你的 origin 逻辑在这里
                if unique_key in triple2origins:
                    fused_results[unique_key]["origins"].update(triple2origins[unique_key])
                
                fused_results[unique_key]["support"] += 1

        except Exception as e:
            # logging.warning(f"Error processing concept '{concept}': {e}")
            continue

    # ---------------------------------------------------------------------
    # 6. 写入结果
    # ---------------------------------------------------------------------
    logging.info(f"Writing {len(fused_results)} fused triples to {output_file}...")
    output_stream = open(output_file, 'w')
    for idx, fused in enumerate(fused_results.values()):
        out = {
            "id": idx,
            "s": fused["s"],
            "p": fused["p"],
            "o": fused["o"],
            "origins": sorted(list(fused["origins"])),
            "support": fused["support"]
        }
        output_stream.write(json.dumps(out, ensure_ascii=False) + "\n")
    output_stream.close()
            
    logging.info("Step 3: Fusion completed.")
