import unittest

import pytest

pytestmark = pytest.mark.heavy
import tempfile
import os
from fastapi.testclient import TestClient
from api.main import app, index_manager

class TestAPI(unittest.TestCase):
    """
    API接口测试
    """
    
    def setUp(self):
        """
        测试前准备
        """
        # 清空索引，保证测试互相隔离（否则 /query 会触发真实 LLM 调用）
        index_manager.reset()
        self.client = TestClient(app)
        # 创建临时目录用于测试
        self.test_dir = tempfile.mkdtemp()
        
    def test_root_endpoint(self):
        """
        测试根路径端点
        """
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("version", data)
        self.assertIn("description", data)
    
    def test_upload_document_success(self):
        """
        测试上传文档端点（成功上传）
        """
        test_file_content = b"This is a test document for upload."
        test_file_name = "test_upload.txt"
        
        with open(os.path.join(self.test_dir, test_file_name), "wb") as f:
            f.write(test_file_content)
        
        with open(os.path.join(self.test_dir, test_file_name), "rb") as f:
            response = self.client.post(
                "/upload",
                files={"file": (test_file_name, f, "text/plain")}
            )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertEqual(data["message"], "文档上传并索引成功")
        self.assertIn("file_id", data)
        self.assertIn("filename", data)
        self.assertEqual(data["filename"], test_file_name)
        self.assertIn("documents_count", data)
        self.assertGreater(data["documents_count"], 0)

    def test_upload_document_invalid_method(self):
        """
        测试上传文档端点（无效方法）
        """
        response = self.client.get("/upload")
        # 应该返回405方法不允许
        self.assertEqual(response.status_code, 405)
    
    def test_query_documents_empty_query(self):
        """
        测试查询文档端点（空查询）
        """
        response = self.client.get("/query")
        # 应该返回422验证错误，因为q参数是必需的
        self.assertEqual(response.status_code, 422)

    def test_topics_endpoint(self):
        """测试专题列表端点"""
        response = self.client.get("/topics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("topics", data)
        self.assertIn("total", data)
        self.assertIsInstance(data["topics"], list)
    
    def test_query_documents_with_query(self):
        """
        测试查询文档端点（带查询参数）
        """
        response = self.client.get("/query?q=测试查询")
        self.assertIn(response.status_code, (200, 400))
        data = response.json()
        if response.status_code == 200:
            self.assertIn("query", data)
            self.assertIn("response", data)
            self.assertIn("source_nodes", data)
        else:
            self.assertIn("detail", data)
    
    def test_get_document_not_found(self):
        """
        测试获取文档端点（文档未找到）
        """
        response = self.client.get("/docs/nonexistent")
        # 应该返回404错误
        self.assertEqual(response.status_code, 404)
    
    def tearDown(self):
        """
        测试后清理
        """
        # 清理临时目录
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()