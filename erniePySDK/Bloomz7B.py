from erniePySDK import getAccessToken,asyncGetAccessToken
import httpx
from typing import Any, AsyncIterator, List, Dict, Iterator
import json

class Bloomz7B:
    def __init__(
        self,
        apiKey: str,
        secretKey: str,
    ):
        self.apiKey = apiKey
        self.secretKey = secretKey
        self.modelUrl = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/bloomz_7b1"

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        access_token = getAccessToken(apiKey=self.apiKey, secretKey=self.secretKey)
        url = self.modelUrl + "?access_token=" + access_token

        payload = json.dumps(
            {
                "messages": messages,
                "stream": stream,
            }
        )
        headers = {"Content-Type": "application/json"}

        if stream:
            with httpx.Client() as client:
                with client.stream(
                    method="POST", url=url, data=payload, headers=headers
                ) as resp:
                    for line in resp.iter_lines():
                        if line:
                            data = line.split("data: ")[1]
                            yield json.loads(data)

        else:
            with httpx.Client() as client:
                resp = client.post(url=url, data=payload, headers=headers,timeout=60)
                yield resp.json()


    async def asyncChat(
            self,
            messages: List[Dict[str, str]],
            stream: bool = False,
    ) -> AsyncIterator[Dict[str,Any]]:
        access_token = await asyncGetAccessToken(apiKey=self.apiKey, secretKey=self.secretKey)
        url = self.modelUrl + "?access_token=" + access_token

        payload = json.dumps(
            {
                "messages": messages,
                "stream": stream,
            }
        )

        headers = {"Content-Type": "application/json"}

        if stream:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    method="POST", url=url, data=payload, headers=headers
                ) as resp:
                    async for line in resp.content.iter_lines():
                        if line:
                            data = line.split("data: ")[1]
                            yield json.loads(data)
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url=url, data=payload, headers=headers, timeout=60)
                yield resp.json()