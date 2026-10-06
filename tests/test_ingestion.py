import os
import tempfile
import unittest

from core import ingestion


class TestIngestion(unittest.TestCase):
    """统一解析与稳定元数据"""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.topic_dir = os.path.join(self.root, "mNGS")
        os.makedirs(self.topic_dir)

    def _write(self, name, content):
        path = os.path.join(self.topic_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_derive_doc_id_and_topic(self):
        path = os.path.join(self.topic_dir, "宏基因组.md")
        self.assertEqual(ingestion.derive_doc_id(path, self.root), "mNGS/宏基因组")
        self.assertEqual(ingestion.derive_topic(path, self.root), "mNGS")

    def test_load_document_sets_stable_metadata(self):
        path = self._write("宏基因组.md", "# mNGS\n\n正文内容。" * 10)
        doc = ingestion.load_document(path, docs_root=self.root, source="indexed")

        self.assertIsNotNone(doc)
        self.assertEqual(doc.id_, "mNGS/宏基因组")
        self.assertEqual(doc.metadata["doc_id"], "mNGS/宏基因组")
        self.assertEqual(doc.metadata["topic"], "mNGS")
        self.assertEqual(doc.metadata["source"], "indexed")
        self.assertEqual(doc.metadata["file_name"], "mNGS/宏基因组.md")
        self.assertTrue(doc.text.strip())

    def test_load_document_upload_uses_given_id(self):
        path = self._write("uploaded.md", "上传文档内容。")
        doc = ingestion.load_document(path, source="uploaded", file_name="上传.md", doc_id="uuid-123")
        self.assertEqual(doc.id_, "uuid-123")
        self.assertEqual(doc.metadata["file_name"], "上传.md")
        self.assertEqual(doc.metadata["source"], "uploaded")
        self.assertEqual(doc.metadata["topic"], "uploads")

    def test_load_documents_from_dir(self):
        self._write("a.md", "内容 A")
        self._write("b.md", "内容 B")
        docs = ingestion.load_documents_from_dir(self.root)
        self.assertEqual(len(docs), 2)
        self.assertEqual({d.metadata["doc_id"] for d in docs}, {"mNGS/a", "mNGS/b"})

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
