import os
from openai import OpenAI
import urllib.parse
import base64
import certifi
import ssl
import httpx

# print(completion.choices[0].message)
hunyuan_key = os.environ.get("HUNYUAN_KEY")

# image_url = "https://img-home.csdnimg.cn/images/20201124032511.png"
# image_description = "This is an image of a bear."
# ark_key = os.environ.get("ARK_API_KEY")
ark_key = 'a42acc36-88c8-4bfb-8ca8-030d80584eee'

client = OpenAI(
    api_key=ark_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # http_client=httpx.Client(verify=False),
)


def encode_image(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# image_path = "./test.png"
# base64_image = encode_image(image_path)

# response = client.chat.completions.create(
#     model="ep-20250311215502-qh2xx",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "图中是什么物体"},
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/png;base64,{base64_image}"},
#                 },
#             ],
#         }
#     ],
#     stream=False,
# )

# print(response.choices[0])
# print("content: ", response.choices[0].message.content)


image_path = "/home/huanganran/field/data/remote.png"
base64_image = encode_image(image_path)

messages = [
    {
        "role": "system",
        "content": (
            "你是一个机器人高层任务规划器。"
            "请将复杂动作拆解为原子动作，每个原子动作必须是一个JSON对象，"
            "格式为: {object, part, action}。"
            "例如: 'type hello' 必须拆解为 ["
            "{object:'keyboard', part:'H', action:'press'}, "
            "{object:'keyboard', part:'E', action:'press'}, ...]。"
        )
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "请用图像中的遥控器打开电视，调到37台，并调大声音。"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            },
        ],
    },
]

response = client.chat.completions.create(
    model="ep-20250311215502-qh2xx",
    messages=messages,
    stream=False,
)

plan = response.choices[0].message.content
print("高层任务计划 JSON: ", plan)
