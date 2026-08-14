"""切分策略测试: token 滑窗、overlap、字段完整性。"""

from fusionrag.chunking import chunking_by_token_size
from fusionrag.utils import Tokenizer


def test_single_chunk_when_short():
    tokenizer = Tokenizer()
    chunks = chunking_by_token_size("短文本", tokenizer, 100, 10)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "短文本"
    assert chunks[0]["chunk_order_index"] == 0
    assert chunks[0]["tokens"] > 0


def test_sliding_window_with_overlap():
    tokenizer = Tokenizer()
    # 足够长的文本, 保证 tiktoken / 字符级两种 tokenizer 下都会切多片
    text = " ".join(f"token{i}" for i in range(500))
    size, overlap = 100, 20
    chunks = chunking_by_token_size(text, tokenizer, size, overlap)
    assert len(chunks) > 2
    assert all(c["tokens"] <= size for c in chunks)
    assert [c["chunk_order_index"] for c in chunks] == list(range(len(chunks)))
    # overlap: 后一片的开头内容出现在前一片中 (token 级重叠)
    assert chunks[1]["content"][:5] in chunks[0]["content"]


def test_overlap_must_be_smaller_than_size():
    tokenizer = Tokenizer()
    try:
        chunking_by_token_size("abc", tokenizer, 10, 10)
        assert False, "应当抛出 ValueError"
    except ValueError:
        pass
