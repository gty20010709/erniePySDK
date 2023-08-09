from erniePySDK import ErnieBot

class ErnieBotTurbo(ErnieBot):
    def __init__(
        self,
        apiKey: str,
        secretKey: str,
    ):
        super().__init__(apiKey, secretKey)
        self.modelUrl = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant"