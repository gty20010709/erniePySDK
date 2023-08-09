import erniePySDK
import os
import asyncio

api_key = os.environ.get("ERNIE_API_KEY","")
secret_key = os.environ.get("ERNIE_SECRET_KEY","")

def testGetAccessToken():
    s = erniePySDK.getAccessToken(apiKey=api_key, secretKey=secret_key)
    assert type(s) == str


async def testAsyncGetAccessToken():
    s = await erniePySDK.asyncGetAccessToken(apiKey=api_key, secretKey=secret_key)
    assert type(s) == str


def testErnieBotChat():
    bot = erniePySDK.ErnieBot(apiKey=api_key, secretKey=secret_key)
    messages = [
            {
                "role": "user",
                "content": "介绍一下你自己"
            }
        ]
    r = next(bot.chat(messages=messages))
    print(f"Result Type: {type(r)}")
    print(r)

def testErnieBotChatStream():
    bot = erniePySDK.ErnieBot(apiKey=api_key, secretKey=secret_key)
    messages = [
            {
                "role": "user",
                "content": "请用Python写一个冒泡排序"
            }
        ]
    chuncks = bot.chat(messages=messages, stream=True)
    for chunck in chuncks:
        print(f"Result Type: {type(chunck)}")
        print(chunck)


async def testErnieBotAsyncChat():
    bot = erniePySDK.ErnieBot(apiKey=api_key, secretKey=secret_key)
    messages = [
            {
                "role": "user",
                "content": "介绍一下你自己"
            }
        ]
    r = next(bot.chat(messages=messages))
    print(f"Result Type: {type(r)}")
    print(r)

async def testErnieBotAsyncChatStream():
    bot = erniePySDK.ErnieBot(apiKey=api_key, secretKey=secret_key)
    messages = [
            {
                "role": "user",
                "content": "Python中的生成器可以在异步程序中使用吗？"
            }
        ]
    chuncks = bot.chat(messages=messages, stream=True)
    for chunck in chuncks:
        # print(f"Result Type: {type(chunck)}")
        print(chunck.get("result"),end="")


def testErnieBotTurbo():
    bot = erniePySDK.ErnieBotTurbo(apiKey=api_key, secretKey=secret_key)
    message = {
        "role": "user",
        "content": "Python中，子类继承父类后如何修改父类的属性？"
    }

    r = next(bot.chat(messages=[message]))
    print(f"Result Type: {type(r)}")
    print(r)


def testBloomz7B():
    bot = erniePySDK.Bloomz7B(apiKey=api_key, secretKey=secret_key)
    message = {
        "role": "user",
        "content": "请介绍你自己"
    }

    r = next(bot.chat(messages=[message]))
    print(f"Result Type: {type(r)}")
    print(r)

def testEmbeddingV1():
    bot = erniePySDK.EmbeddingV1(apiKey=api_key, secretKey=secret_key)
    texts = [
        "请介绍你自己",
        "Python中，子类继承父类后如何修改父类的属性？",
        "什么是词向量?"
    ]

    r = bot.embedding(texts=texts)
    print(r)


async def testAsyncEmbeddingV1():
    bot = erniePySDK.EmbeddingV1(apiKey=api_key, secretKey=secret_key)
    texts = [
        "请介绍你自己",
        "Python中，子类继承父类后如何修改父类的属性？",
        "什么是词向量?"
    ]

    r = await bot.asyncEmbedding(texts=texts)
    print(r)

if __name__ == "__main__":
    asyncio.run(testAsyncEmbeddingV1())
