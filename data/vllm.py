import requests
import json
from typing import List, Iterable, Tuple
from omegaconf.listconfig import ListConfig

from general_util.logger import get_child_logger

logger = get_child_logger("VLLM")


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      max_tokens: int = 16,
                      temperature: float = 0.0,
                      use_beam_search: bool = False,
                      stream: bool = False,
                      stop: List[str] = ["</s>"],
                      **kwargs) -> requests.Response:
    headers = {"User-Agent": "MERIt Test Client"}
    p_load = {
        "prompt": prompt,
        "n": n,
        "use_beam_search": use_beam_search,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "stop": stop,
    }
    p_load.update(kwargs)
    response = requests.post(api_url, headers=headers, json=p_load, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


def get_response(response: requests.Response) -> Tuple[str, str]:
    data = json.loads(response.content)
    output = data["text"]
    return output, data


class VLLMRequestGenerator:
    def __init__(self, api_url: str, n: int = 1, max_tokens: int = 1024, use_beam_search: bool = False, stream: bool = False,
                 temperature: float = 0.0, stop: List[str] = ["</s>"], **kwargs):
        self.api_url = api_url
        self.n = n
        self.max_tokens = max_tokens
        self.use_beam_search = use_beam_search
        self.stream = stream
        self.stop = stop
        self.temperature = temperature
        self.kwargs = kwargs

    def __call__(self, prompt: str) -> str:
        if "completions" not in self.api_url:
            response, content = get_response(post_http_request(prompt,
                                                               self.api_url,
                                                               n=self.n,
                                                               max_tokens=self.max_tokens,
                                                               temperature=self.temperature,
                                                               use_beam_search=self.use_beam_search,
                                                               stream=self.stream,
                                                               stop=self.stop,
                                                               **self.kwargs))[0]
            response = response.replace(prompt, "")
        else:
            response = post_http_request(prompt,
                                         self.api_url,
                                         n=self.n,
                                         max_tokens=self.max_tokens,
                                         temperature=self.temperature,
                                         use_beam_search=self.use_beam_search,
                                         stream=self.stream,
                                         stop=self.stop,
                                         **self.kwargs)
            # print(json.loads(response.content))
            # content = json.loads(response.content)
            if response.status_code != 200:
                logger.warning(response.content)
                response = ""
            else:
                outputs = []
                for item in json.loads(response.content)["choices"]:
                    outputs.append(item["text"].replace(prompt, ""))
                if len(outputs) == 1:
                    response = outputs[0]
                else:
                    response = outputs

        return response
