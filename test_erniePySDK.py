# 在运行 pytest 时，会发生一个 warning ：
# ...
# erniePySDK\.venv\Lib\site-packages\httpx\_content.py:204: DeprecationWarning: Use 'content=<...>' to upload raw bytes/text content.
# ...
# 
# 这个锅当由百度官方的API接口来背: 
# 向 API 发送 post 请求时，需要提交 data， 在 httpx 中，data 应该时 Dict[str,Any] 格式
# 但官方接口要求 str 格式的，所以需要 json.dumps(Dict[str,Any]) 
# 如果传入 Dict[str,Any] 格式的数据，官方接口会报错： {'error_code': 336002, 'error_msg': 'Invalid JSON', 'id': 'as-kg400kq1hk'}
# 这是警告的来源，可以可以忽视

import erniePySDK
import os
import pytest

api_key = os.environ.get("ERNIE_API_KEY","")
secret_key = os.environ.get("ERNIE_SECRET_KEY","")

def test_getAccessToken():
    s = erniePySDK.getAccessToken(apiKey=api_key, secretKey=secret_key)
    assert type(s) == str

@pytest.mark.asyncio
async def test_asyncGetAccessToken():
    s = await erniePySDK.asyncGetAccessToken(apiKey=api_key, secretKey=secret_key)
    assert type(s) == str


def test_ErnieBotChat():
    bot = erniePySDK.ErnieBot(apiKey=api_key, secretKey=secret_key)
    messages = [
            {
                "role": "user",
                "content": "介绍一下你自己"
            }
        ]
    r = next(bot.chat(messages=messages))
    assert type(r) == dict

def test_ErnieBotChatStream():
    bot = erniePySDK.ErnieBot(apiKey=api_key, secretKey=secret_key)
    messages = [
            {
                "role": "user",
                "content": "请用Python写一个冒泡排序"
            }
        ]
    chunks = bot.chat(messages=messages, stream=True)
    for chunk in chunks:
        assert type(chunk) == dict

@pytest.mark.asyncio
async def test_ErnieBotAsyncChat():
    bot = erniePySDK.ErnieBot(apiKey=api_key, secretKey=secret_key)
    messages = [
            {
                "role": "user",
                "content": "介绍一下你自己"
            }
        ]
    r = next(bot.chat(messages=messages))
    assert type(r) == dict

@pytest.mark.asyncio
async def test_ErnieBotAsyncChatStream():
    bot = erniePySDK.ErnieBot(apiKey=api_key, secretKey=secret_key)
    messages = [
            {
                "role": "user",
                "content": "Python中的生成器可以在异步程序中使用吗？"
            }
        ]
    chunks = bot.chat(messages=messages, stream=True)
    for chunk in chunks:
        assert type(chunk) == dict


# ErnieBotTurbo类 完全继承 ErnieBot类，只是模型的请求地址（URL）不同
# 所以， ErnieBot 能通过的测试，ErnieBotTurbo 也能通过，不需要重复测试
def test_ErnieBotTurbo():
    bot = erniePySDK.ErnieBotTurbo(apiKey=api_key, secretKey=secret_key)
    message = {
        "role": "user",
        "content": "Python中，子类继承父类后如何修改父类的属性？"
    }

    r = next(bot.chat(messages=[message]))
    assert type(r) == dict


def test_Bloomz7B():
    bot = erniePySDK.Bloomz7B(apiKey=api_key, secretKey=secret_key)
    message = [{
        "role": "user",
        "content": "北京有什么山？"
        }
    ]

    r = next(bot.chat(messages=message))
    assert type(r) == dict

def test_Bloomz7BStream():
    bot = erniePySDK.Bloomz7B(apiKey=api_key, secretKey=secret_key)
    message = [{
        "role": "user",
        "content": "北京有什么山？"
        }
    ]
    chunks = bot.chat(messages=message, stream=True)
    for chunk in chunks:
        assert type(chunk) == dict

@pytest.mark.asyncio
async def test_Bloomz7BAsync():
    bot = erniePySDK.Bloomz7B(apiKey=api_key, secretKey=secret_key)
    message = [{
        "role": "user",
        "content": "北京有什么山？"
        }
    ]
    r = next(bot.chat(messages=message))
    assert type(r) == dict

@pytest.mark.asyncio
async def test_Bloomz7BAsyncStream():
    bot = erniePySDK.Bloomz7B(apiKey=api_key, secretKey=secret_key)
    message = [{
        "role": "user",
        "content": "北京有什么山？"
        }
    ]
    chunks = bot.chat(messages=message, stream=True)
    for chunk in chunks:
        assert type(chunk) == dict


def test_embeddingV1():
    bot = erniePySDK.EmbeddingV1(apiKey=api_key, secretKey=secret_key)
    texts = [
        "请介绍你自己",
        "Python中，子类继承父类后如何修改父类的属性？",
        "什么是词向量?"
    ]

    r = bot.embedding(texts=texts)
    assert type(r) == dict

@pytest.mark.asyncio
async def test_asyncEmbeddingV1():
    bot = erniePySDK.EmbeddingV1(apiKey=api_key, secretKey=secret_key)
    texts = [
        "请介绍你自己",
        "Python中，子类继承父类后如何修改父类的属性？",
        "什么是词向量?"
    ]

    r = await bot.asyncEmbedding(texts=texts)
    assert type(r) == dict

