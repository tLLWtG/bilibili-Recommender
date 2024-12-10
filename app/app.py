import time, os
from urllib.parse import urlparse
import requests
import qrcode
import base64
from PIL import Image
from io import BytesIO
from flask import Flask, render_template, request, jsonify, redirect, url_for

from app.getHistoryData import get_history_data
from app.getHotData import get_hot_data
from app.getRecommandData import get_recommand_data


# app = Flask(__name__)

# bilibili 二维码登陆相关的 api
QR_CODE_GENERATE_URL = (
    "https://passport.bilibili.com/x/passport-login/web/qrcode/generate"
)
QR_CODE_POLL_URL = "https://passport.bilibili.com/x/passport-login/web/qrcode/poll"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}

cookie_file_path = "user_data/cookie.txt"
cookie_data = {}
cookie_str = ""

img_path = "app/static/user_img"
img_path_rel = "static/user_img"


# 申请二维码 url
def get_qrcodekey():
    response = requests.get(QR_CODE_GENERATE_URL, headers=headers)
    print(response)
    if response.status_code == 200:
        data = response.json()
        if data["code"] == 0:
            return data["data"]["url"] + "main-fe-header", data["data"]["qrcode_key"]
        else:
            return None, None
    return None, None


# url 转二维码图片
def generate_qrcode_base64(_url):
    url = _url
    if url:
        # 创建二维码
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)

        img = qr.make_image(fill="black", back_color="white")
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        # 将图片转为 Base64 编码
        qr_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return qr_base64
    return None


def check_qrcode_status(qrcode_key):
    global cookie_data, cookie_str
    response = requests.get(
        QR_CODE_POLL_URL, params={"qrcode_key": qrcode_key}, headers=headers
    )

    if response.status_code == 200:
        data = response.json()
        code = data["data"]["code"]
        print(data)
        if code == 0:
            # 登录成功，返回 cookies 和其他数据
            cookie_data = response.cookies.get_dict()
            print(cookie_data)
            # 将 cookies 转换为字符串
            cookie_str = "; ".join(
                [f"{key}={value}" for key, value in cookie_data.items()]
            )
            # add buvid3 for hotdata
            cookie_str = "buvid3=1; " + cookie_str
            print(cookie_str)
            timestamp = data["data"]["timestamp"]
            url = data["data"]["url"]
            return code, timestamp, url, cookie_str
        else:
            # 其他状态
            return code, None, None, None
    return None, None, None, None


def download_img(image_url):
    if not os.path.exists(img_path):
        try:
            os.makedirs(img_path)
            print(f"目录 {img_path} 创建成功")
        except Exception as e:
            print(f"创建目录时出错: {e}")
    # 使用 urlparse 解析 URL
    parsed_url = urlparse(image_url)
    file_name = os.path.basename(parsed_url.path)
    response = requests.get(image_url, headers=headers)
    if response.status_code == 200:
        with open(os.path.join(img_path, file_name), "wb") as file:
            file.write(response.content)
        print(f"图片下载成功，保存为 {os.path.join(img_path, file_name)}")
    else:
        print("图片下载失败，状态码:", response.status_code)


def get_history_info():
    history_info = get_history_data(cookie_str, 10)
    for info in history_info:
        download_img(info["pic"])
    return history_info


def get_hot_info():
    hot_info = get_hot_data(cookie_str, 10)
    for info in hot_info:
        download_img(info["pic"])
    return hot_info


def get_explore_info():
    explore_info = get_recommand_data(cookie_str, 10)
    for info in explore_info:
        download_img(info["pic"])
    return explore_info


# 首页路由
# @app.route("/")
def home():
    # 判断是否已有 cookie，然后跳转对应界面
    global cookie_str, headers
    if os.path.exists(cookie_file_path):
        with open(cookie_file_path, "r") as f:
            cookie_str = f.read()
            headers["Cookie"] = cookie_str
            print(f"read:{cookie_str}")
            # return redirect(url_for("dashboard"))
            return dashboard()
    else:
        # return redirect(url_for("login"))
        return login()


# login 接口
# @app.route("/qrcode_status", methods=["GET"])
def qrcode_status():
    global headers
    qrcode_key = request.args.get("qrcode_key")
    if not qrcode_key:
        return jsonify({"error": "Missing qrcode_key"}), 400

    status, timestamp, url, cookies = check_qrcode_status(qrcode_key)
    if status == 86101:
        print("未扫描")
        return jsonify({"status": "not scanned"}), 200
    elif status == 86038:
        print("二维码失效")
        return jsonify({"error": "QR code expired"}), 400
    elif status == 86090:
        print("已扫描未确认")
        return jsonify({"status": "scanned but not confirmed"}), 200
    elif status == 0:
        print("登录成功")
        print(cookies)

        # 确保目录存在
        cookie_dir = os.path.dirname(cookie_file_path)
        if not os.path.exists(cookie_dir):
            try:
                os.makedirs(cookie_dir)
                print(f"目录 {cookie_dir} 创建成功")
            except Exception as e:
                print(f"创建目录时出错: {e}")
        # 写入 cookie.txt 文件
        try:
            with open(cookie_file_path, "w") as f:
                f.write(f"{cookies}")
            print(f"SESSDATA 已成功保存到 {cookie_file_path}")
            headers["Cookie"] = cookies
        except Exception as e:
            print(f"保存 SESSDATA 时出错: {e}")
        return (
            jsonify(
                {
                    "status": "login success",
                    "timestamp": timestamp,
                    "url": url,
                    "cookies": cookies,
                }
            ),
            200,
        )


# @app.route("/login")
def login():
    url, qrcode_key = get_qrcodekey()
    print(url)
    qr_base64 = generate_qrcode_base64(url)
    if url and qrcode_key:
        return render_template("login.html", qr_code=qr_base64, qrcode_key=qrcode_key)
    return "Error generating QR code."


# @app.route("/dashboard")
def dashboard():
    global cookie_str
    headers["Cookie"] = cookie_str
    history_info = get_history_info()
    for x in history_info:
        parsed_url = urlparse(x["pic"])
        file_name = os.path.basename(parsed_url.path)
        full_path = os.path.join(img_path_rel, file_name)
        x["pic"] = os.path.normpath(full_path).replace("\\", "/")
        temp = []
        count = 0
        for xx in x["tag"]:
            if len(xx) <= 8 and count < 2:  # 标签长度不超过 8 且最多取 2 个标签
                temp.append(xx)
                count += 1
        x["tag"] = temp
    return render_template(
        "dashboard.html", cookie_str=cookie_str, history_info=history_info
    )


# @app.route("/logout", methods=["POST"])
def logout():
    # 清除保存的 cookies
    if os.path.exists(cookie_file_path):
        os.remove(cookie_file_path)
    return jsonify({"success": True})


# @app.route("/api/recommend-hot-vid", methods=["GET"])
def recommend_hot_vid():
    res = get_hot_info()
    for x in res:
        parsed_url = urlparse(x["pic"])
        file_name = os.path.basename(parsed_url.path)
        full_path = os.path.join(img_path_rel, file_name)
        x["pic"] = os.path.normpath(full_path).replace("\\", "/")
    return jsonify(res)


# @app.route("/api/recommend-explore-vid", methods=["GET"])
def recommend_explore_vid():
    res = get_explore_info()
    for x in res:
        parsed_url = urlparse(x["pic"])
        file_name = os.path.basename(parsed_url.path)
        full_path = os.path.join(img_path_rel, file_name)
        x["pic"] = os.path.normpath(full_path).replace("\\", "/")
    return jsonify(res)
