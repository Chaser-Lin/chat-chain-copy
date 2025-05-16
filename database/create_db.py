import os
import sys
import re
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tempfile
from dotenv import load_dotenv, find_dotenv
from embedding.call_embedding import get_embedding
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma
# 首先实现基本配置

DEFAULT_DB_PATH = "./knowledge_db"
DEFAULT_PERSIST_PATH = "./vector_db"

# 获取路径下的文件列表
def get_files(dir_path):
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            # os.path.join 获取文件的绝对路径？
            file_list.append(os.path.join(filepath, filename))
    return file_list


def file_loader(file, loaders):
    # 这个是什么意思？
    if isinstance(file, tempfile._TemporaryFileWrapper):
        file = file.name
    # 如果不是文件，则递归加载目录下的文件
    if not os.path.isfile(file):
        [file_loader(os.path.join(file, f), loaders) for f in  os.listdir(file)]
        return
    file_type = file.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file)) # pdf加载器
    elif file_type == 'md':
        pattern = r"不存在|风控"
        match = re.search(pattern, file)
        # 正则匹配没风险的文件才会加载
        if not match:
            loaders.append(UnstructuredMarkdownLoader(file)) # md加载器
    elif file_type == 'txt':
        loaders.append(UnstructuredFileLoader(file))
    return

# 根据传入数据和embedding模型创建向量数据库，过滤非法的模型传参
def create_db_info(files=DEFAULT_DB_PATH, embeddings="openai", persist_directory=DEFAULT_PERSIST_PATH):
    if embeddings == 'openai' or embeddings == 'm3e' or embeddings =='zhipuai':
        vectordb = create_db(files, persist_directory, embeddings)
    return ""


def create_db(files=DEFAULT_DB_PATH, persist_directory=DEFAULT_PERSIST_PATH, embeddings="openai"):
    """
    该函数用于加载 PDF 文件，切分文档，生成文档的嵌入向量，创建向量数据库。

    参数:
    file: 存放文件的路径。
    embeddings: 用于生产 Embedding 的模型

    返回:
    vectordb: 创建的数据库。
    """
    if files == None:
        return "can't load empty file"
    if type(files) != list:
        files = [files]
    loaders = []
    [file_loader(file, loaders) for file in files]
    docs = []
    for loader in loaders:
        if loader is not None:
            docs.extend(loader.load())
    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)
    # 根据embedding名称获取本地封装的embedding模型实例
    if type(embeddings) == str:
        embeddings = get_embedding(embedding=embeddings)
    # 定义持久化路径
    persist_directory = './vector_db/chroma'
    # 加载数据库
    vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    ) 

    vectordb.persist() # 将数据持久化到磁盘
    return vectordb


def presit_knowledge_db(vectordb):
    """
    该函数用于持久化向量数据库。

    参数:
    vectordb: 要持久化的向量数据库。
    """
    vectordb.persist()


def load_knowledge_db(path, embeddings):
    """
    该函数用于加载向量数据库。

    参数:
    path: 要加载的向量数据库路径。
    embeddings: 向量数据库使用的 embedding 模型。

    返回:
    vectordb: 加载的数据库。
    """
    vectordb = Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )
    return vectordb

# 只有当该文件作为主程序入口时才会执行，如果文件作为模块导入，则不会执行
if __name__ == "__main__":
    create_db(embeddings="m3e")
