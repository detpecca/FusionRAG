"""实体消歧: 抽取产出的实体名归一到规范名, 解决"字节跳动/字节跳动公司/ByteDance"
成为三个孤立节点的问题。

流程 (在 merge 之前执行):
1. 精确别名表命中: alias_kv(alias -> canonical) 直接改名, 零 LLM 成本
2. embedding 相似度预筛: 新实体名 vs 存量实体名 (及新实体两两之间),
   余弦相似度 >= 阈值的对进入候选
3. LLM 确认: 候选对是否为同一实体的别名/全称/缩写 (走 LLM 缓存,
   同一对名字跨文档导入只确认一次)
4. 重写 nodes/edges: 别名记录并入规范名 (description/账本由既有 merge
   逻辑自然合并), 映射写回 alias_kv

注意: 删除路径的描述重建 (_rebuild_surviving_descriptions) 按规范名
精确查找重抽结果, 别名命中的记录会走"未命中"兜底 (保留原描述),
属于可接受的降级, 不影响删除一致性。
"""

from __future__ import annotations

import logging

import numpy as np

from .llm import LLMService
from .prompts import PROMPTS
from .storage import BaseKVStorage
from .utils import cosine_similarity

logger = logging.getLogger("fusionrag.disambiguate")


async def resolve_entity_aliases(
    nodes: dict[str, list[dict]],
    edges: dict[tuple[str, str], list[dict]],
    existing_names: list[str],
    alias_kv: BaseKVStorage,
    llm: LLMService,
    config,
) -> tuple[dict[str, list[dict]], dict[tuple[str, str], list[dict]], dict[str, str]]:
    """把 nodes/edges 中的别名实体名归一到规范名。

    返回 (重写后的 nodes, 重写后的 edges, 本次新增的 alias->canonical 映射)。
    """
    if not config.entity_disambiguation or len(nodes) < 1:
        return nodes, edges, {}

    mapping: dict[str, str] = {}

    # ---- 1) 持久别名表精确命中 (零 LLM 成本) ----
    known = await alias_kv.get_by_ids(list(nodes.keys()))
    for name, rec in zip(nodes.keys(), known):
        if rec and rec.get("canonical"):
            mapping[name] = rec["canonical"]

    # ---- 2) embedding 相似度预筛 ----
    remaining = [n for n in nodes if n not in mapping]
    # 存量实体中与本批重名的直接自然合并, 不进消歧
    existing = [n for n in existing_names if n not in nodes]
    candidates: list[tuple[str, str]] = []  # (alias, canonical)
    if remaining:
        names_to_embed = remaining + [n for n in existing if n]
        if len(names_to_embed) > 1:
            vectors = await llm.embed(names_to_embed)
            n_rem = len(remaining)
            sims = cosine_similarity(
                np.asarray(vectors[:n_rem], dtype=np.float32),
                np.asarray(vectors, dtype=np.float32),
            )
            thr = config.entity_alias_similarity_threshold
            # 新名 -> 存量名 (取相似度最高且达阈值的)
            for i, alias in enumerate(remaining):
                row = sims[i]
                best_j, best_v = -1, thr
                for j in range(n_rem, len(names_to_embed)):
                    if float(row[j]) > best_v:
                        best_j, best_v = j, float(row[j])
                if best_j >= 0:
                    candidates.append((alias, names_to_embed[best_j]))
            # 新名两两之间 (同批别名, 规范名取字典序较小者, 保证确定性)
            for i in range(n_rem):
                for j in range(i + 1, n_rem):
                    if float(sims[i][j]) >= thr:
                        a, b = sorted((remaining[i], remaining[j]))
                        candidates.append((b, a))

    # ---- 3) LLM 确认 (走缓存: 同一对名字只花一次钱) ----
    for alias, canonical in candidates:
        if alias in mapping or canonical in mapping:
            # 已有映射的不再重复确认 (含传递: a->b 且 b->c 时只保留 a->c 的结果)
            continue
        if await _confirm_same_entity(llm, alias, canonical):
            mapping[alias] = canonical

    # 传递解析: a->b 而 b 本身又是别名 -> 指向最终规范名
    def _resolve(name: str) -> str:
        seen = set()
        while name in mapping and name not in seen:
            seen.add(name)
            name = mapping[name]
        return name

    for alias in list(mapping):
        mapping[alias] = _resolve(mapping[alias])

    if not mapping:
        return nodes, edges, {}

    # ---- 4) 重写 nodes / edges ----
    new_nodes: dict[str, list[dict]] = {}
    for name, records in nodes.items():
        canon = _resolve(name)
        new_nodes.setdefault(canon, []).extend(records)

    new_edges: dict[tuple[str, str], list[dict]] = {}
    for (src, tgt), records in edges.items():
        s, t = _resolve(src), _resolve(tgt)
        if s == t:
            continue  # 消歧后两端同一实体的自环边丢弃
        new_edges.setdefault((s, t), []).extend(records)

    # ---- 5) 映射持久化 (下次导入直接走精确命中) ----
    await alias_kv.upsert({a: {"canonical": c} for a, c in mapping.items()})

    logger.info(
        "实体消歧: %d 个别名归一 (%s)",
        len(mapping),
        ", ".join(f"{a}->{c}" for a, c in mapping.items()),
    )
    return new_nodes, new_edges, mapping


async def _confirm_same_entity(llm: LLMService, alias: str, canonical: str) -> bool:
    """LLM 确认两个名字是否指同一实体, 输出限定 YES/NO。"""
    prompt = f"Name A: {alias}\nName B: {canonical}\nAnswer:"
    try:
        answer = await llm.chat(
            prompt,
            system_prompt=PROMPTS["entity_alias_check"],
            use_cache=True,
            cache_type="alias",
        )
    except Exception:
        logger.warning("别名确认 LLM 调用失败 (%s / %s), 保守处理为不同实体", alias, canonical)
        return False
    return str(answer).strip().upper().startswith("YES")
