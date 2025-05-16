from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """`Zhipuai Embeddings` embedding models."""

    zhipuai_api_key: Optional[str] = None
    """Zhipuai application apikey"""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict: 
        """
        Validate whether zhipuai_api_key in the environment variables or
        configuration file are available or not.

        Args:
            values是包含配置信息的dict必须包含 zhipuai_api_key 这个值
            values: a dictionary containing configuration information, must include the
            fields of zhipuai_api_key
        Returns:
            如果传进的values或者环境变量中都没有 zhipuai_api_key 的值, 则会返回原有values变量
            a dictionary containing configuration information. If zhipuai_api_key
            are not provided in the environment variables or configuration
            file, the original values will be returned; otherwise, values containing
            zhipuai_api_key will be returned.
        Raises:

            ValueError: zhipuai package not found, please install it with `pip install
            zhipuai`
        """
        # 从字典或者环境变量中找到对应kv
        values["zhipuai_api_key"] = get_from_dict_or_env(
            values,
            "zhipuai_api_key",
            "ZHIPUAI_API_KEY",
        )

        try:
            import zhipuai
            zhipuai.api_key = values["zhipuai_api_key"]
            values["client"] = zhipuai.model_api
        # import依赖不存在报错
        except ImportError:
            raise ValueError(
                "Zhipuai package not found, please install it with "
                "`pip install zhipuai`"
            )
        return values

    # 调用embedding模型获取文本向量信息
    def _embed(self, texts: str) -> List[float]:
        # send request
        try:
            resp = self.client.invoke(
                model="text_embedding",
                prompt=texts
            )
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        if resp["code"] != 200:
            raise ValueError(
                "Error raised by inference API HTTP code: %s, %s"
                % (resp["code"], resp["msg"])
            )
        embeddings = resp["data"]["embedding"]
        return embeddings

    # 获取查询文本的向量信息
    def embed_query(self, text: str) -> List[float]:
        """
        Embedding a text.

        Args:

            Text (str): A text to be embedded.

        Return:

            List [float]: An embedding list of input text, which is a list of floating-point values.
        """
        resp = self.embed_documents([text])
        return resp[0]

    # 把文本列表转换成向量列表
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of text documents.

        Args:
            texts (List[str]): A list of text documents to embed.

        Returns:
            List[List[float]]: A list of embeddings for each document in the input list.
                            Each embedding is represented as a list of float values.
        """
        return [self._embed(text) for text in texts]

    # 异步embedding方法，直接不提供
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError(
            "Please use `embed_documents`. Official does not support asynchronous requests")

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError(
            "Please use `aembed_query`. Official does not support asynchronous requests")
