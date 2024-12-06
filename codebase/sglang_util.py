import requests
from llm_template import paraphrase_template, keyword_template, text_expansion_template
import os


# https://github.com/sgl-project/sglang/blob/2b0fc5941d3d7f3dfe4a56c053ddddf9d4f77670/docs/backend/backend.md
def get_paraphase_response(client, model_name, query_input, generation_config):
    text2paraphrase = paraphrase_template.format(query_input)

    params = dict(
        # temperature=generation_config['temperature'],
        max_tokens=generation_config['max_tokens'],
        # top_p=generation_config['top_p'],
        timeout=generation_config['timeout'],
        # do_sample = True
    )

    msg = [
        {
            "role": "user",
            "content": text2paraphrase
        }
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=msg,
        **params
    )

    return response.choices[0].message.content.strip()


def get_keyword_response(client, model_name, query_input, generation_config):
    ext2parse = keyword_template.format(query_input)

    params = dict(
        # temperature=generation_config['temperature'],
        max_tokens=generation_config['max_tokens'],
        # top_p=generation_config['top_p'],
        timeout=generation_config['timeout'],
        # do_sample=True
    )

    msg = [
        {
            "role": "user",
            "content": ext2parse
        }
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=msg,
        **params
    )

    return response.choices[0].message.content.strip()


def get_text_expansion_response(client, model_name, kw_list_to_expand, generation_config):
    response_list = dict()
    for kw in kw_list_to_expand:
        text_expan = text_expansion_template.format(kw)
        params = dict(
            # temperature=generation_config['temperature'],
            max_tokens=generation_config['max_tokens'],
            # top_p=generation_config['top_p'],
            timeout=generation_config['timeout'],
            # do_sample=True
        )

        msg = [
            {
                "role": "user",
                "content": text_expan
            }
        ]
        response = client.chat.completions.create(
            model=model_name,
            messages=msg,
            **params
        )
        response_list[response.choices[0].message.content.strip()] = kw
    return response_list


def main():
    url = "http://localhost:31000/v1/chat/completions"

    # sglang_paraphrase_model_inference()
    data = {
        "model": "/media/zilun/wd-161/hf_download/Llama-3.2-3B-Instruct",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    }

    response = requests.post(url, json=data)
    print(response.json())
    # print_highlight(response.json())


if __name__ == "__main__":
    main()
