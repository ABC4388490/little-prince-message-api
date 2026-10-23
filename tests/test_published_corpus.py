"""Release data checks: source alignment and paraphrase/interpretation boundaries."""
import json
from pathlib import Path
from unittest.mock import patch

from app.services.agent_service import decide_strategy
from rag.retriever.session import lexical_candidates
from scripts.build_chunk_meta import build_passage_chunks, build_analysis_chunks

RAG = Path(__file__).resolve().parents[1] / 'rag'

def records():
    return json.loads((RAG/'chunk_meta.json').read_text())

def test_release_contains_all_191_notes_with_stable_unique_ids_and_source_links():
    rows = records()
    assert len(rows) == 191
    assert len({r['chunk_id'] for r in rows}) == 191
    assert all(r['source_url'].startswith('https://gutenberg.net.au/') for r in rows)
    assert all('【原文】' not in r['content'] for r in rows)
    assert all('梗概' in r['content'] or '阅读解读' in r['content'] for r in rows)

def test_chapters_match_the_original_story_order_and_switchman_occupation():
    rows = {r['chunk_id']: r for r in records()}
    for chapter, needle in [(4,'B612'), (8,'玫瑰'), (9,'离开'), (17,'蛇'), (18,'花'), (19,'高山'), (20,'玫瑰园'), (22,'扳道工'), (23,'解渴药丸')]:
        assert needle in rows[f'passage_{chapter:02d}_01']['chapter']
    assert '调度列车' in rows['passage_22_01']['content']
    assert '不是拧螺丝' in rows['passage_22_01']['content']
    assert '不是第十三章' in rows['passage_23_05']['content']

def test_published_data_is_reproducible_from_reviewed_sources():
    chapters = json.loads((RAG/'sources/chapters.json').read_text())
    analysis = json.loads((RAG/'sources/analysis.json').read_text())
    rebuilt = build_passage_chunks(chapters) + build_analysis_chunks(analysis)
    expected = records()
    assert [(r['chunk_id'],r['content'],r['chapter']) for r in rebuilt] == [(r['chunk_id'],r['content'],r['chapter']) for r in expected]
    assert json.loads((RAG/'data.json').read_text())['chunks'] == expected

def test_expanded_material_is_available_without_loading_vector_models(monkeypatch):
    monkeypatch.setenv('RAG_RETRIEVAL_MODE','lexical')
    with patch('rag.retriever.session.get_retriever', side_effect=AssertionError('No ML model')):
        rows=lexical_candidates('扳道工调度列车')
    assert any(r['chunk_id'].startswith('passage_22_') for r in rows)
    assert decide_strategy('扳道工在干什么？').use_rag

def test_expanded_story_questions_retrieve_but_daily_outings_do_not():
    for text in ['你在沙漠遇到的那朵三片花瓣的小花，跟你说了什么？',
                 '你到了高山上，听见了什么？', '解渴药丸能省下多少时间？',
                 '玫瑰园里有多少玫瑰？']:
        assert decide_strategy(text).use_rag, text
    for text in ['我今天去公园看花了', '周末想去爬高山', '我在沙漠旅游时听到了风声']:
        assert not decide_strategy(text).use_rag, text
