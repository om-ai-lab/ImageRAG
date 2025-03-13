import json
import os
import math
from tqdm import tqdm


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_mmerealworld(question_fpath):
    with open(question_fpath, 'r') as file:
        questions = json.load(file)
    questions = [question for question in questions if question["Subtask"] == "Remote Sensing"]
    questions = get_chunk(questions, 1, 0)
    return questions
    
    
def get_question_content(mmerealworld_questions):
    return [line["Question_id"] for line in mmerealworld_questions], [line["Image"] for line in mmerealworld_questions]
    
    
def main():
    mmerealworld_json = "/data1/zilun/ImageRAG0226/codebase/inference/MME-RealWorld-RS/MME_RealWorld.json"
    mmerealworld_img = "/data9/shz/dataset/MME-RealWorld/remote_sensing"

    mmerealworldlite_json = "/data1/zilun/ImageRAG0226/codebase/inference/MME-RealWorld-RS/MME-RealWorld-Lite-toi_corrected.json"
    mmerealworldlite_img = "/data9/shz/dataset/MME-RealWorld/remote_sensing/remote_sensing"

    mmerealworld_rs_img = "/data1/zilun/MME-RealWorld-RS"
    mmerealworld_todo_json = "/data1/zilun/ImageRAG0226/codebase/inference/MME-RealWorld-RS/MME-RealWorld-excludelite.json"
    
    mmerealworldlite_json = load_mmerealworld(mmerealworldlite_json)
    mmerealworld_json = load_mmerealworld(mmerealworld_json)
    print(len(mmerealworldlite_json))
    print(len(mmerealworld_json))

    question_ids_mmerealworld, img_names_mmerealworld = get_question_content(mmerealworld_json)
    question_ids_mmerealworldlite, img_names_mmerealworldlite = get_question_content(mmerealworldlite_json)
    
    exclude_question_ids = []
    include_question_ids = []
    for question_id_mmerealworld in tqdm(question_ids_mmerealworld):
        # revised_question_id_mmerealworld = question_id_mmerealworld.split("/")[-1]
        if question_id_mmerealworld not in question_ids_mmerealworldlite:
            exclude_question_ids.append(question_id_mmerealworld)
        else:
            include_question_ids.append(question_id_mmerealworld)
    print(len(exclude_question_ids))
    print(len(include_question_ids))

    save_questions = []
    for question_id_mmerealworld in tqdm(exclude_question_ids):
        for line in mmerealworld_json:
            if line["Question_id"] == question_id_mmerealworld:
                save_questions.append(line)
    print(len(save_questions))
    print(save_questions[:3])
    
    with open(mmerealworld_todo_json, "w") as f:
        json.dump(save_questions, f, indent=4)


if __name__ == "__main__":
    main()