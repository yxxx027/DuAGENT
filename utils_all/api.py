# utils_all/api.py
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

class APIClient:
    def __init__(self, api_key ,url, api_name ):
        self.api_key  = api_key 
        self.url = url 
        self.api_name = api_name
        self.client = OpenAI(api_key=api_key, base_url=url)

    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(10))
    def request_gpt(self, textprompt):
        response = self.client.chat.completions.create(
            model=self.api_name,
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": textprompt}]
            }],
            temperature=0.7,
            top_p=0.8
        )
        return response.choices[0].message.content

    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(10))
    def request_gpt_with_image(self, textprompt, img_type, img_b64_str):
        response = self.client.chat.completions.create(
            model=self.api_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": textprompt},
                        {"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"}}
                    ],
                }
            ],
            temperature=0.7,
            top_p=0.8,
        )
        return response.choices[0].message.content
