import requests
import selenium
import json

# len为需要读取的信息量（有可能会小于len,因为可能总共的信息就不多）


def get_hot_data(cookie, len):
    # 目标 URL
    url = "https://api.bilibili.com/x/web-interface/popular/series/list"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Cookie": cookie,
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        videos_info = response.json()
        if videos_info["code"] != 0:
            print(videos_info["message"])
            return
    else:
        print(f"请求失败，状态码：{response.status_code}")
    # 获取最新一期的数据，page即为期数
    page = videos_info["data"]["list"][0]["number"]
    # res为最终的结果
    res = []
    count = 1

    while count <= len:
        url = "https://api.bilibili.com/x/web-interface/popular/series/one?number="
        str_page = str(page)
        url += str_page
        print(url)
        # 解析 JSON 数据
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            videos_info = response.json()
            if videos_info["code"] != 0:
                print(videos_info["message"])
                return
        else:
            print(f"请求失败，状态码：{response.status_code}")

        for video_info in videos_info["data"]["list"]:
            sigle_res = {}
            sigle_res["bvid"] = video_info["bvid"]
            sigle_res["title"] = video_info["title"]
            sigle_res["pic"] = video_info["pic"]
            sigle_res["author"] = video_info["owner"]["name"]
            sigle_res["view"] = video_info["stat"]["view"]
            sigle_res["like"] = video_info["stat"]["like"]
            sigle_res["favorite"] = video_info["stat"]["favorite"]
            sigle_res["coin"] = video_info["stat"]["coin"]
            sigle_res["share"] = video_info["stat"]["share"]
            # tag比较麻烦，需要单独去获取详细信息
            url_2 = (
                "https://api.bilibili.com/x/web-interface/view/detail?bvid="
                + video_info["bvid"]
            )
            response = requests.get(url_2, headers=headers)
            if response.status_code == 200:
                video_detail = response.json()
                if video_detail["code"] != 0:
                    print(video_detail["message"])
                    return
            else:
                print(f"请求失败，状态码：{response.status_code}")
            # print(video_detail['data']['Tags'])
            sigle_res["tag"] = [tag["tag_name"] for tag in video_detail["data"]["Tags"]]

            res.append(sigle_res)
            print(video_info["bvid"], count)
            count += 1
            if count > len:
                break
        page -= 1
        if page == 0:
            break
        with open("hotvideo.json", "w", encoding="utf-8") as json_file:
            # 使用 json.dump() 将字典写入文件
            json.dump(
                res, json_file, indent=4, ensure_ascii=False
            )  # indent=4 用来让输出格式更易读

    return res
