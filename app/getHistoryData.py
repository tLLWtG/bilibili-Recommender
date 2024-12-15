import requests
import json

from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_session_with_retries(retries=5, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_fav_data(user_mid, headers, maxcnt=20):
    """
    获取用户所有收藏夹中的视频信息
    """
    all_videos = []
    session = get_session_with_retries()
    folders_url = "https://api.bilibili.com/x/v3/fav/folder/created/list-all"
    params = {
        "up_mid": user_mid,
        "jsonp": "jsonp"
    }
    try:
        response = session.get(folders_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"获取收藏夹列表失败：{e}")
        return None

    folders_info = response.json()
    if folders_info['code'] != 0:
        print(f"获取收藏夹列表失败，错误信息：{folders_info.get('message', '')}")
        return None

    folders = folders_info['data']['list']
    print(f"找到 {len(folders)} 个收藏夹")

    for folder in folders:
        folder_id = folder['id']
        folder_title = folder['title']
        print(f"正在获取收藏夹 '{folder_title}' (ID: {folder_id}) 的视频")
        
        page_num = 1
        page_size = 20  # 每页的视频数量，B站通常为20
        while True:
            resources_url = "https://api.bilibili.com/x/v3/fav/resource/list"
            params = {
                "media_id": folder_id,
                "pn": page_num,
                "ps": page_size,
                "keyword": "",
                "order": "mtime",
                "jsonp": "jsonp"
            }
            try:
                response = session.get(resources_url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"获取收藏夹 '{folder_title}' 视频失败：{e}")
                break

            resources_info = response.json()
            if resources_info['code'] != 0:
                print(f"获取收藏夹 '{folder_title}' 视频失败，错误信息：{resources_info.get('message', '')}")
                break

            resources = resources_info['data']['medias']
            if not resources:
                print(f"收藏夹 '{folder_title}' 没有更多视频")
                break
            for media in resources:
                video = {
                    "bvid": media['bvid'],
                    "pic": media['cover'],
                    "author": media['upper']['name'],
                    "view": media['cnt_info']['play'],
                    "like": None,       # 这个api里没有，有也是0
                    "favorite": None,  
                    "coin": media['cnt_info']['collect'],
                    "share": None,     
                    "duration": None,
                    "progress": None,
                    "tag": [],
                    "isfaved": 1,
                    "isliked": 0 # 这api里面没有，神秘了，当没有吧
                }
                detail_info_url = "https://api.bilibili.com/x/web-interface/view/detail?bvid=" + video['bvid']
                try:
                    response = session.get(detail_info_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    video_detail = response.json()
                    if video_detail['code'] != 0:
                        print(video_detail['message'])
                        print(detail_info_url)
                        continue
                except requests.exceptions.RequestException as e:
                    print(f"请求详情信息失败：{e}")
                    continue
                
                # print(video_detail['data'])
                
                video['duration'] = video_detail['data']['View']['duration']
                video['progress'] = video['duration'] / 2       # 这个api里没有，取个均值意思一下
                video['tag'] = [tag['tag_name'] for tag in video_detail['data']['Tags']]
                video['like'] = video_detail['data']['View']['stat']['like']
                video['reply'] = video_detail['data']['View']['stat']['reply']
                video['favorite'] = video_detail['data']['View']['stat']['favorite']
                video['share'] = video_detail['data']['View']['stat']['share']
                
                all_videos.append(video)
                
                if maxcnt is not None and len(all_videos) >= maxcnt:
                    return all_videos
            print(f"已获取收藏夹 '{folder_title}' 的第 {page_num} 页视频")
            if resources_info['data']['has_more'] == 0:
                break
            page_num += 1

    return all_videos

def get_vote_data(cookie, headers=None):
    """
    获取用户的评分信息，包括收藏、点赞和投币的视频
    （后面两个API有问题，获取收藏就行）
    """
    if headers is None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Cookie": cookie
        }
    
    session = get_session_with_retries()
    
    try:
        response = session.get("https://api.bilibili.com/x/web-interface/nav", headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"请求导航信息失败：{e}")
        return None
    
    user_info = response.json()
    if user_info['code'] != 0:
        print(f"获取用户信息失败，错误信息：{user_info.get('message', '')}")
        return None
    
    user_mid = user_info['data']['mid']
    print(f"用户mid: {user_mid}")
    
    fav_data = get_fav_data(user_mid, headers)
    # like_data = get_like_data(user_mid, headers)
    # coin_data = get_coin_data(user_mid, headers)
    
    return fav_data


# cookie为用户的相关信息， history_len为需要读取的信息量（有可能会小于history_len,因为可能总共的信息就不多 ）

def get_history_data(cookie, history_len):  # [改了个变量名因为和len()冲突了]
    # 目标 URL
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Cookie": cookie 
    }
    #res为最终的结果
    res = []
    count = 1
    #page是用来翻页的
    page = 1
    
    res = res + get_vote_data(cookie)
    print(f"已获取{len(res)}条收藏历史记录")
    
    while(count <= history_len):
        url = "https://api.bilibili.com/x/v2/history?pn="
        str_page = str(page)
        url += str_page
        # 解析 JSON 数据
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            videos_info = response.json()
            if(videos_info['code'] != 0):
                print(videos_info['message'])
                print(url)
                continue
        else:
            print(f"请求失败，状态码：{response.status_code}")
        
        for video_info in videos_info['data'] :        
            sigle_res = {}
            sigle_res["bvid"] = video_info['bvid']
            sigle_res["pic"] = video_info["pic"]
            sigle_res['author'] = video_info['owner']['name']
            sigle_res["view"] = video_info["stat"]['view']
            sigle_res["like"] = video_info["stat"]['like']
            sigle_res["favorite"] = video_info["stat"]['favorite']
            sigle_res["coin"] = video_info["stat"]['coin']
            sigle_res["share"] = video_info["stat"]['share']
                
            # [加入时长和进度]
            sigle_res["duration"] = video_info["duration"]
            sigle_res["progress"] = video_info["progress"]
            if video_info["progress"] == -1:    # -1好像是看完了？
                sigle_res["progress"] = sigle_res["duration"]
                
            # tag比价麻烦，需要单独去获取详细信息
            url_2 = "https://api.bilibili.com/x/web-interface/view/detail?bvid=" + video_info['bvid']
            response = requests.get(url_2, headers=headers)
            if response.status_code == 200:
                video_detail = response.json()
                if(video_detail['code'] != 0):
                    print(video_detail['message'])
                    print(url_2)
                    continue
            else:
                print(f"请求失败，状态码：{response.status_code}")      
            #print(video_detail['data']['Tags'])
            sigle_res['tag'] = [tag['tag_name'] for tag in video_detail['data']['Tags']]
                
            # [加入是否被点赞和收藏]
            sigle_res['isfaved'] = 1 if video_info['favorite'] else 0
            url_2 = f"https://api.bilibili.com/x/web-interface/archive/has/like?aid={video_info['stat']['aid']}"
            response = requests.get(url_2, headers=headers)
            if response.status_code == 200:
                video_detail = response.json()
                if(video_detail['code'] != 0):
                    print(video_detail['message'])
                    print(url_2)
                    continue      
            else:
                print(f"请求失败，状态码：{response.status_code}")      
            sigle_res['isliked'] = video_detail['data']
            

            res.append(sigle_res)
            # print(video_info['bvid'], count)
            count+=1
            print(f"已获取{count}条历史记录")
            if(count > history_len):
                break
        page+=1     
        
        with open('historyVideo.json', 'w', encoding='utf-8') as json_file:
        # 使用 json.dump() 将字典写入文件
            json.dump(res, json_file, indent=4,ensure_ascii=False)  # indent=4 用来让输出格式更易读   
    
    return res




