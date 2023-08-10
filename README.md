# 文心千帆 Python SDK

本库为文心千帆 Python SDK, 非官方库 ———— 目前官方还没有 Python 的 SDK。

[文心千帆](https://cloud.baidu.com/product/wenxinworkshop)

目前支持：

* ERNIE-Bot 
* ERNIE-Bot-turbo
* BLOOMZ-7B
* Embeddings

### 安装

**PyPI地址：[https://pypi.org/project/erniepysdk/](https://pypi.org/project/erniepysdk/)**

```bash

pip install erniePySDK

```

### 使用示例
```py
# ErnieBot 流式对话
# ErnieBotTurbo 和 ErnieBot 的调用方法，甚至参数都完全相同
# Bloozm7B 和 ErnieBot 的调用方法也完全相同，只是参数不同，少了 top_p 等几个参数
import erniePySDK

api_key = ""
secret_key = ""

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

if __name__ == "__main__":
    testErnieBotChatStream()

```


### 获取文心千帆 APIKEY
1. 在百度云官网进行申请：https://cloud.baidu.com/product/wenxinworkshop
2. 申请通过后创建应用：https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application
3. 获取 apikey 和 api secret

### 其他示例
<details>
<summary>ErnieBot 对话 </summary>

```py
import erniePySDK

api_key = ""
secret_key = ""

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


testErnieBotChat()

```
</details>


<details>
<summary>ErnieBot 对话 异步调用</summary>

```py
import erniePySDK
import asyncio

api_key = ""
secret_key = ""

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

asyncio.run(testErnieBotAsyncChat())

```
</details>




<details>
<summary>ErnitBot 流式对话 异步调用</summary>

```py
import erniePySDK
import asyncio

api_key = ""
secret_key = ""

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


asyncio.run(testErnieBotAsyncChatStream())
```
</details>


<details>
<summary>EmbeddingV1 调用 </summary>

```python
import erniePySDK

apiKey = ""
secretKey = ""

def testEmbeddingV1():
    bot = erniePySDK.EmbeddingV1(apiKey=api_key, secretKey=secret_key)
    texts = [
        "请介绍你自己",
        "Python中，子类继承父类后如何修改父类的属性？",
        "什么是词向量?"
    ]

    r = bot.embedding(texts=texts)
    print(r)

testEmbeddingV1()
```

</details>

<details>
<summary>EmbeddingV1 异步调用 </summary>

```py
import erniePySDK
import asyncio

apiKey = ""
secretKey = ""

async def testAsyncEmbeddingV1():
    bot = erniePySDK.EmbeddingV1(apiKey=api_key, secretKey=secret_key)
    texts = [
        "请介绍你自己",
        "Python中，子类继承父类后如何修改父类的属性？",
        "什么是词向量?"
    ]

    r = await bot.asyncEmbedding(texts=texts)
    print(r)

asyncio.run(testAsyncEmbeddingV1())

```


</details>