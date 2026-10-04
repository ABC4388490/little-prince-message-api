import hashlib
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from app import create_app
from app import db


class SQLiteConversationConnection:
    """Local route tests; CI uses real Postgres via TEST_DATABASE_URL."""
    def __init__(self, filename):
        self.conn = sqlite3.connect(filename)

    def execute(self, sql, params=()):
        return self.conn.execute(sql.replace('%s', '?'), params)

    def commit(self):
        self.conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.conn.close()


class TestMemorySecurity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.filename = str(Path(self.temp.name)/'memory.db')
        self.addCleanup(patch.stopall)
        self.postgres_url = os.environ.get('TEST_DATABASE_URL', '')
        patch.dict(os.environ, {'DATABASE_URL': self.postgres_url}).start()
        patch('app.db.db_path', return_value=self.filename).start()
        db.init_databases()
        if not self.postgres_url:
            with closing(sqlite3.connect(self.filename)) as conn:
                conn.executescript('''
                    CREATE TABLE conversations (
                        id INTEGER PRIMARY KEY, visitor_id TEXT NOT NULL UNIQUE);
                    CREATE TABLE messages_v2 (
                        id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL,
                        role TEXT, content TEXT, pos_x REAL, pos_y REAL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                ''')
            patch('app.api.conversations.db_url', return_value='local-test-adapter').start()
            patch('app.api.conversations.connect_pg', side_effect=lambda: SQLiteConversationConnection(self.filename)).start()
        self.client = create_app(skip_chat_blueprint=True, skip_db_init=True).test_client()
        self.llm = patch('app.api.conversations.complete_chat', return_value='星星会陪着你。').start()
        self.a, self.b = self.session(), self.session()

    def session(self):
        response = self.client.post('/api/session')
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        body = response.get_json()
        return {'Authorization': 'Bearer ' + body['sessionToken']}

    def conversation(self, headers):
        response = self.client.get('/api/conversations/me', headers=headers)
        self.assertEqual(response.status_code, 200)
        return response.get_json()['conversationId']

    def connection(self):
        return db.connect_pg() if self.postgres_url else SQLiteConversationConnection(self.filename)

    def test_missing_forged_and_legacy_credentials_rejected(self):
        for headers in [{}, {'X-Visitor-Id':'someone'}, {'Authorization':'Bearer '+('A'*43)}]:
            for url in ['/api/conversations/me?visitorId=someone', '/api/conversations/1/messages', '/api/profile?visitorId=someone']:
                self.assertEqual(self.client.get(url, headers=headers).status_code, 401)
            for url in ['/api/profile', '/api/conversations/1/messages']:
                self.assertEqual(self.client.post(url, headers=headers, json={'visitorId':'someone'}).status_code, 401)
        self.llm.assert_not_called()

    def test_sessions_cannot_claim_old_ids(self):
        self.assertEqual(self.client.post('/api/session', json={'visitorId':'old'}).status_code, 400)
        self.assertEqual(self.client.post('/api/session?visitorId=old').status_code, 400)

    def test_tokens_are_stored_as_hashes_and_expire(self):
        token = self.a['Authorization'].split()[1]
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with self.connection() as conn:
            row = conn.execute('SELECT token_hash FROM visitor_sessions WHERE token_hash = %s', (token_hash,)).fetchone()
            self.assertEqual(row[0], token_hash)
            self.assertNotEqual(row[0], token)
            conn.execute('UPDATE visitor_sessions SET expires_at = 0 WHERE token_hash = %s', (token_hash,))
            conn.commit()
        self.assertEqual(self.client.get('/api/profile', headers=self.a).status_code, 401)

    def test_foreign_conversation_read_and_write_rejected_before_llm(self):
        cid = self.conversation(self.a)
        payload = {'messages':[{'role':'user','content':'只有我能看到这句话'}]}
        for method in ['get', 'post']:
            response = getattr(self.client, method)(f'/api/conversations/{cid}/messages', headers=self.b, json=payload)
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.llm.assert_not_called()

    def test_owner_roundtrip_and_identity_spoofing_ignored(self):
        cid = self.conversation(self.a)
        self.assertEqual(self.conversation(self.a), cid)
        self.assertNotEqual(self.conversation(self.b), cid)
        response = self.client.post(f'/api/conversations/{cid}/messages', headers=self.a,
                                    json={'messages':[{'role':'user','content':'你好'}], 'visitorId':'forged'})
        self.assertEqual(response.status_code, 201)
        history = self.client.get(f'/api/conversations/{cid}/messages', headers=self.a)
        self.assertEqual([m['content'] for m in history.get_json()], ['你好', '星星会陪着你。'])
        self.assertEqual(history.headers['Cache-Control'], 'no-store')

    def test_profile_roundtrip_and_isolation(self):
        payload = {'visitorId':'victim', 'emotion':'low', 'worries':['工作'], 'roundCount':3}
        response = self.client.post('/api/profile', headers=self.a, json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get('/api/profile?visitorId=victim', headers=self.a).get_json()['roundCount'], 3)
        self.assertEqual(self.client.get('/api/profile?visitorId=victim', headers=self.b).get_json()['roundCount'], 0)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')

    def test_database_failure_does_not_bypass_authentication(self):
        target = 'app.auth.connect_pg' if self.postgres_url else 'app.auth.connect_sqlite'
        with patch(target, side_effect=RuntimeError('database unavailable')):
            self.assertEqual(self.client.get('/api/profile', headers=self.a).status_code, 503)
            self.assertEqual(self.client.get('/api/conversations/me', headers=self.a).status_code, 503)
        self.llm.assert_not_called()

    def test_preflight_allows_authorization(self):
        response = self.client.options('/api/profile', headers={
            'Origin':'https://little-prince.xyz', 'Access-Control-Request-Method':'GET',
            'Access-Control-Request-Headers':'Authorization'})
        self.assertIn('authorization', response.headers.get('Access-Control-Allow-Headers', '').lower())


if __name__ == '__main__':
    unittest.main()
