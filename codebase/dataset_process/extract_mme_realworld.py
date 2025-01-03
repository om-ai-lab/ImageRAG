import json
import os


def extract_dota_image_names(json_file_path):
    """
    从JSON文件中提取所有dota相关的图片名称，文件名以P开头。

    参数:
        json_file_path (str): JSON文件的路径。

    返回:
        list: 包含所有符合条件的图片文件名的列表。
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"文件未找到: {json_file_path}")
        return []
    except json.JSONDecodeError:
        print(f"无法解析JSON文件: {json_file_path}")
        return []

    count = 0
    for d in data:
        if d["Subtask"] == "Remote Sensing":
            count += 1
    print(count)

    dota_image_names = []
    potsdam_image_names = []
    for entry in data:
        input_image = entry.get("input_image", "")
        base_name = os.path.basename(input_image)
        if "dota" in input_image:
            dota_name = base_name.split("_")[-1]
            dota_image_names.append(dota_name)
        if "potsdam" in input_image:
            # remote_sensing/top_potsdam_3_12_RGB_Potsdam.png
            potsdam_name = input_image.split("_Potsdam.png")[0].split("/")[-1]
            potsdam_image_names.append(potsdam_name)

    return dota_image_names, potsdam_image_names

def main():
    # 请将 'data.json' 替换为你的JSON文件路径
    json_file = '/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/annotation_mme-realworld.json'
    dota2_dir = '/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/dota2_test'
    all_dota2_file = [d.split('.')[0] for d in os.listdir(dota2_dir)]
    dota_image_names, potsdam_image_names = extract_dota_image_names(json_file)
    print(len(dota_image_names), len(potsdam_image_names))
    print(len(dota_image_names) + len(potsdam_image_names))
    count = 0
    for dota_image_name in dota_image_names:
        if dota_image_name.split(".")[0] in all_dota2_file:
            count += 1
        else:
            print(dota_image_name)
    print(count, len(dota_image_names))
    # # 输出结果
    # print("提取到的Dota图片文件名:")
    # for img in images:
    #     print(img)
    #
    # # 如果需要，将结果保存到一个文本文件
    # with open('/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/mme-realworld_dota_image_names.txt', 'w', encoding='utf-8') as out_file:
    #     for img in images:
    #         out_file.write(f"{img}\n")
    #
    # print("图片文件名已保存到 '/media/zilun/fanxiang4t/GRSM/ImageRAG_git/data/mme-realworld_dota_image_names.txt'")


if __name__ == "__main__":
    main()
