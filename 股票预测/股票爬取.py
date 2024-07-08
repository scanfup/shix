import csv
from curl_cffi import requests as cffi_requests
import json

# 读取JSON文件
with open("url_ids.json", "r", encoding="utf-8") as file:
    url_ids = json.load(file)  # 使用json.load()方法解析JSON数据

# 请求头
headers = {
    "accept": "*/*",
    "accept-language": "zh-CN,zh;q=0.9",
    "content-type": "application/json",
    "dnt": "1",
    "domain-id": "cn",
    "origin": "https://cn.investing.com",
    "priority": "u=1, i",
    "referer": "https://cn.investing.com/",
    "sec-ch-ua": "\"Not/A)Brand\";v=\"8\", \"Chromium\";v=\"126\", \"Google Chrome\";v=\"126\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "sec-gpc": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}

# 定义CSV文件名
csv_filename = 'stock_data.csv'

# 打开CSV文件
with open(csv_filename, mode='w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)

    # 写入表头
    writer.writerow(['Stock_ID', 'Date', 'Close', 'Open', 'High', 'Low', 'Volume'])

    # 循环每个股票ID
    for url_id in url_ids:
        # 构造URL
        url = f"https://api.investing.com/api/financialdata/historical/{url_id}?start-date=2024-05-15&end-date=2024-07-08&time-frame=Daily&add-missing-rows=false"
        params = {
            "start-date": "2014-01-10",
            "end-date": "2024-07-08",
            "time-frame": "Daily",
            "add-missing-rows": "false"
        }

        response = cffi_requests.get(url, impersonate='chrome110', timeout=10, headers=headers, params=params)

        # 检查请求是否成功
        if response.status_code == 200:
            data = response.json()

            # 提取我们关心的字段
            records = data['data']

            # 写入数据
            for record in records:
                date = record['rowDateTimestamp']
                close = record['last_close']
                open_ = record['last_open']
                high = record['last_max']
                low = record['last_min']
                volume = record['volume']

                writer.writerow([url_id, date, close, open_, high, low, volume])

            print(f"Data for stock {url_id} saved to {csv_filename}")
        else:
            print(f"Failed to retrieve data for stock {url_id}: {response.status_code}")

print("All data has been saved.")
