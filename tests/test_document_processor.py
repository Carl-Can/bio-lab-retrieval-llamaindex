import os
import tempfile
import unittest

from core.document_processor import DocumentProcessor


class TestDocumentProcessor(unittest.TestCase):
    """文档处理（统一走 ingestion + SentenceSplitter）"""

    def setUp(self):
        self.doc_processor = DocumentProcessor()
        self.test_dir = tempfile.mkdtemp()

    def _write(self, name, content):
        path = os.path.join(self.test_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_process_document_returns_chunks_with_metadata(self):
        path = self._write("test.md", "# 标题\n\n这是一段用于测试的正文内容。" * 5)
        chunks = self.doc_processor.process_document(path)

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn("chunk_id", chunk)
            self.assertIn("text", chunk)
            self.assertIn("metadata", chunk)
        self.assertEqual(chunks[0]["metadata"]["doc_id"], "test")

    def test_get_indexed_chunks_without_index(self):
        self.assertEqual(self.doc_processor.get_indexed_chunks("whatever"), [])

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
