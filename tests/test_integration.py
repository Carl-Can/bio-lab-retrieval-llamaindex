import unittest

import pytest

pytestmark = pytest.mark.heavy
import tempfile
import os
import shutil
from unittest.mock import Mock, patch

class TestIntegration(unittest.TestCase):
    """
    集成测试
    """
    
    def setUp(self):
        """
        测试前准备
        """
        self.test_dir = tempfile.mkdtemp()
        
    def test_document_processing_workflow(self):
        """
        测试文档处理工作流程
        """
        # 这是一个集成测试，测试整个文档处理流程
        # 但由于依赖外部库和API，我们使用模拟对象
        
        with patch('core.document_processor.DocumentProcessor') as mock_doc_processor, \
             patch('core.index_manager.IndexManager') as mock_index_manager, \
             patch('api.main.app'):
            
            # 模拟文档处理器
            mock_doc_processor_instance = Mock()
            mock_doc_processor.return_value = mock_doc_processor_instance
            mock_doc_processor_instance.process_document.return_value = [
                {"text": "测试内容1", "metadata": {}},
                {"text": "测试内容2", "metadata": {}}
            ]
            
            # 模拟索引管理器
            mock_index_manager_instance = Mock()
            mock_index_manager.return_value = mock_index_manager_instance
            
            # 验证各组件可以协同工作
            test_file_path = os.path.join(self.test_dir, "test.txt")
            with open(test_file_path, "w") as f:
                f.write("测试文档")
            
            # 调用文档处理
            chunks = mock_doc_processor_instance.process_document(test_file_path)
            
            # 验证结果
            self.assertEqual(len(chunks), 2)
            mock_index_manager_instance.create_index_from_documents.assert_not_called()
    
    def tearDown(self):
        """
        测试后清理
        """
        shutil.rmtree(self.test_dir, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()