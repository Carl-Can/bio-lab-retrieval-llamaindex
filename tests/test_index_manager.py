import unittest
from unittest.mock import Mock, patch

from core.index_manager import IndexManager


def _fake_chroma_client(count=0):
    client = Mock()
    collection = Mock()
    collection.count.return_value = count
    client.get_or_create_collection.return_value = collection
    client.delete_collection.return_value = None
    return client


class TestIndexManager(unittest.TestCase):
    """索引管理器单元测试（不加载真实模型 / 不连真实 Chroma）"""

    def setUp(self):
        patcher = patch.multiple(
            "core.index_manager",
            OpenAI=Mock(),
            OpenAIEmbedding=Mock(),
            HuggingFaceEmbedding=Mock(),
            Settings=Mock(),
            ChromaVectorStore=Mock(),
            StorageContext=Mock(),
            VectorStoreIndex=Mock(),
        )
        self.addCleanup(patcher.stop)
        patcher.start()

        client_patcher = patch(
            "core.index_manager.chromadb.PersistentClient",
            return_value=_fake_chroma_client(count=0),
        )
        self.addCleanup(client_patcher.stop)
        client_patcher.start()

        self.index_manager = IndexManager()

    def test_initialization(self):
        self.assertIsNotNone(self.index_manager.llm)
        self.assertIsNotNone(self.index_manager.embed_model)
        self.assertIsNone(self.index_manager.index)

    @patch("core.index_manager.VectorStoreIndex")
    def test_create_index_from_documents(self, mock_index):
        mock_index_instance = Mock()
        mock_index.from_documents.return_value = mock_index_instance

        result = self.index_manager.create_index_from_documents([Mock(), Mock()])

        self.assertEqual(result, mock_index_instance)
        mock_index.from_documents.assert_called_once()

    def test_get_query_engine_requires_index(self):
        with self.assertRaises(ValueError):
            self.index_manager.get_query_engine()

    def test_retrieve_requires_index(self):
        with self.assertRaises(ValueError):
            self.index_manager.retrieve("查询", top_k=3)

    @patch("core.index_manager.RetrieverQueryEngine")
    @patch("core.index_manager.build_retriever")
    def test_query_index(self, mock_build_retriever, mock_rqe):
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="测试响应")
        mock_engine = Mock()
        mock_engine.query.return_value = mock_response
        mock_rqe.from_args.return_value = mock_engine
        self.index_manager.index = Mock()

        result = self.index_manager.query_index("测试查询")

        self.assertEqual(str(result), "测试响应")
        mock_engine.query.assert_called_once_with("测试查询")
        mock_build_retriever.assert_called_once()

    @patch("core.index_manager.build_reranker", return_value=None)
    @patch("core.index_manager.build_retriever")
    def test_retrieve_returns_top_k(self, mock_build_retriever, mock_build_reranker):
        fake_nodes = [Mock(), Mock(), Mock(), Mock()]
        mock_build_retriever.return_value.retrieve.return_value = fake_nodes
        self.index_manager.index = Mock()

        nodes = self.index_manager.retrieve("查询", top_k=3)

        self.assertEqual(len(nodes), 3)

    @patch("core.index_manager.VectorStoreIndex")
    def test_delete_document_calls_delete_ref_doc(self, mock_index):
        mock_index_instance = Mock()
        self.index_manager.index = mock_index_instance

        self.index_manager.delete_document("mNGS/foo")

        mock_index_instance.delete_ref_doc.assert_called_once_with(
            "mNGS/foo", delete_from_docstore=True
        )


if __name__ == "__main__":
    unittest.main()
