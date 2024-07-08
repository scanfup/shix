import json

# 读取JSON文件
with open("demo.json", "r", encoding="utf-8") as file:
    response = json.load(file)  # 使用json.load()方法解析JSON数据

# 提取特定字段
url_ids = [i["pair_ID"] for i in response["hits"]]

# 保存提取的字段到新的JSON文件
with open("url_ids.json", "w", encoding="utf-8") as file:
    json.dump(url_ids, file, ensure_ascii=False, indent=4)

print("URL IDs have been saved to url_ids.json")
