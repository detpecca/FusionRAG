"""Prompt 模板。

核心设计:
- 抽取输出采用分隔符格式: 每条记录一行, 字段间用 tuple_delimiter, 结束用 completion_delimiter
- 关系记录带 keywords 字段, 用于关系向量检索
- gleaning 补抽 prompt
- 查询关键词提取(high-level / low-level)
- 图谱上下文 + 文本片段统一组装进 system prompt
"""

GRAPH_FIELD_SEP = "<SEP>"          # 图节点/边字段值内部的分隔符
TUPLE_DELIMITER = "<|#|>"          # 记录内字段分隔符
RECORD_DELIMITER = "\n"            # 记录间分隔符 (简化: 固定换行)
COMPLETION_DELIMITER = "<|COMPLETE|>"

DEFAULT_ENTITY_TYPES = [
    "Person", "Creature", "Organization", "Location", "Event",
    "Concept", "Method", "Content", "Data", "Artifact", "NaturalObject",
]

PROMPTS: dict[str, str] = {}

# ---------------------------------------------------------------------------
# 实体与关系抽取
# ---------------------------------------------------------------------------

PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, classify it as `Other`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities, decompose it into multiple binary (two-entity) relationship pairs for separate description.
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction.
        *   `relationship_keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords within this field must be separated by a comma `,` **(DO NOT use `{tuple_delimiter}`)**.
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities.
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   Incorrect Example: `entity{tuple_delimiter}Tok{tuple_delimiter}location{tuple_delimiter}Tok is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected**; swapping the source and target entities does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Output at most **{max_entity_records}** entities and **{max_total_records}** records in total (entities + relationships), choosing the most meaningful ones.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships have been completely extracted.

---Entity Types---
{entity_types}

---Output Format Template---
{examples}
"""

PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract entities and relationships from the input text below.

---Input Text---
{input_text}

---Output---
"""

PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Output Format:** Follow the exact output format as before (4 fields for entity, 5 fields for relation, delimited by `{tuple_delimiter}`). Do **NOT** output entities and relationships that were previously extracted correctly.

2.  **Completion Signal:** Output the literal string `{completion_delimiter}` when finished.

---Output---
"""

# few-shot 示例 (简化: 一个实体 + 一个关系)
PROMPTS["entity_extraction_examples"] = """Example:
entity{tuple_delimiter}Alex{tuple_delimiter}Person{tuple_delimiter}Alex is a character who experiences frustration with Jordan's dismissive attitude.
relation{tuple_delimiter}Alex{tuple_delimiter}Jordan{tuple_delimiter}conflict, workplace tension{tuple_delimiter}Alex and Jordan share a history of workplace friction regarding communication styles.
{completion_delimiter}
"""

# ---------------------------------------------------------------------------
# 描述合并后的 LLM 摘要
# ---------------------------------------------------------------------------

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON Lines format. Each line (representing a single description) is a JSON object with a `Description` key.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting before or after the text.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full context.
5. Length: The summary should not exceed {summary_length} tokens.

---Input---
{description_list}

---Output---
"""

# ---------------------------------------------------------------------------
# 查询关键词提取
# ---------------------------------------------------------------------------

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a knowledge graph retrieval system.

---Instructions---
1.  **Analyze the Query:** Carefully examine the user's query to understand the core subject and the level of detail required.
2.  **Extract Keywords:** Identify and categorize keywords into two distinct types:
    *   **`high_level_keywords`**: For overarching concepts or themes, capturing the core subject, intent, or situation.
    *   **`low_level_keywords`**: For specific entities, proper nouns, technical jargon, product names, or concrete items.
3.  **Conciseness & Meaningfulness:** Keywords should be concise words or meaningful phrases.
4.  **Language:** All extracted keywords must be written in `{language}`.
5.  **Output Format:** Return a single JSON object. The JSON object must have exactly two keys: `high_level_keywords` and `low_level_keywords`, each containing a list of strings.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
"""

PROMPTS["keywords_extraction_examples"] = """[
  {
    "User Query": "What are the environmental consequences of deforestation on biodiversity?",
    "Output": {"high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"], "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions"]}
  }
]
"""

# ---------------------------------------------------------------------------
# 问答
# ---------------------------------------------------------------------------

PROMPTS["kg_query_context"] = """---Entities---
```json
{entities_str}
```

---Relationships---
```json
{relations_str}
```

---Document Chunks---
```json
{text_chunks_str}
```
"""

PROMPTS["naive_query_context"] = """---Document Chunks---
```json
{text_chunks_str}
```
"""

PROMPTS["rag_response"] = """---Role---
You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---
Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---
1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical answer. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
2. Content & Grounding:
  - Strictly adhere to the provided context for the answer; do NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.
3. Citations:
  - Each chunk in `Document Chunks` carries an `id`. Whenever a statement in your answer draws on one or more chunks, append its citation marker(s) immediately after the statement, in the form [1] (or [1][3] for multiple).
  - Citation markers MUST ONLY reference chunk ids that actually appear in the **Context**; never fabricate a marker.
4. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.
5. Additional Instructions: {user_prompt}

---Context---
{context_data}
"""

PROMPTS["naive_rag_response"] = PROMPTS["rag_response"]

PROMPTS["entity_alias_check"] = """---Role---
You are an Entity Alias Judge for knowledge graph construction.

---Task---
The user message contains two entity names. Decide whether they refer to the
SAME real-world entity (alias, abbreviation, full name, translation, or
spelling variant), or two DIFFERENT entities.

---Instructions---
- Judge only by the names themselves; do not speculate beyond common knowledge
  of well-known organizations/products/people.
- When uncertain, answer NO (keeping entities separate is safe, merging
  different entities is not recoverable).
- Output exactly one line: YES or NO. No explanation.
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question. "
    "The knowledge base does not contain enough relevant information."
)
