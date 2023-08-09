import cachetools
import asyncache
import httpx
from erniePySDK.Errors import ApiKeyOrSecretKeyError


# accessToken 获取周期
# 有效时间是 30 天，提前一天获取
TTL = 29 * 24 * 60 * 60


@cachetools.cached(cache=cachetools.TTLCache(maxsize=1, ttl=TTL))
def getAccessToken(apiKey: str, secretKey: str) -> str:
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": apiKey,
        "client_secret": secretKey,
    }
    with httpx.Client() as client:
        resp = client.post(url=url, params=params)
        try:
            return resp.json()["access_token"]
        except:
            raise ApiKeyOrSecretKeyError("apiKey or secretKey error")


@asyncache.cached(cache=cachetools.TTLCache(maxsize=1, ttl=TTL))
async def asyncGetAccessToken(apiKey: str, secretKey: str) -> str:
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": apiKey,
        "client_secret": secretKey,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url=url, params=params)
        try:
            return resp.json()["access_token"]
        except:
            raise ApiKeyOrSecretKeyError("apiKey or secretKey error")
