from ast import Dict
from typing import List
from erniePySDK import getAccessToken, asyncGetAccessToken
import json
import httpx


class EmbeddingV1:
    def __init__(
        self,
        apiKey: str,
        secretKey: str,
    ):
        self.apiKey = apiKey
        self.secretKey = secretKey
        self.modelUrl = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1"

    def embedding(self, texts: List[str]) -> Dict:
        """
        输入文本以获取embeddings。说明：
            （1）文本数量不超过16
            （2）每个文本长度不超过 384个token
        """

        url = (
            self.modelUrl
            + "?access_token="
            + getAccessToken(apiKey=self.apiKey, secretKey=self.secretKey)
        )
        payload = json.dumps({"input": texts})
        headers = {"Content-Type": "application/json"}
        with httpx.Client() as client:
            resp = client.post(url=url, data=payload, headers=headers, timeout=60)
        return resp.json()

    async def asyncEmbedding(
        self,
        texts: List[str],
    ) -> Dict:
        access_token = await asyncGetAccessToken(
            apiKey=self.apiKey, secretKey=self.secretKey
        )
        url = self.modelUrl + "?access_token=" + access_token
        payload = json.dumps({"input": texts})
        headers = {"Content-Type": "application/json"}
        async with httpx.AsyncClient() as client:
            resp = await client.post(url=url, data=payload, headers=headers, timeout=60)
        return resp.json()
