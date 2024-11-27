import time, os
import requests
import qrcode
import base64
from PIL import Image
from io import BytesIO
from flask import Flask, render_template, request, jsonify, redirect, url_for


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
cookie_data = ""


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
    response = requests.get(
        QR_CODE_POLL_URL, params={"qrcode_key": qrcode_key}, headers=headers
    )

    if response.status_code == 200:
        data = response.json()
        code = data["data"]["code"]
        print(data)
        if code == 0:
            # 登录成功，返回 cookies 和其他数据
            cookies = response.cookies.get_dict()
            timestamp = data["data"]["timestamp"]
            url = data["data"]["url"]
            return code, timestamp, url, cookies
        else:
            # 其他状态
            return code, None, None, None
    return None, None, None, None


# 首页路由
# @app.route("/")
def home():
    # 判断是否已有 cookie，然后跳转对应界面
    if os.path.exists(cookie_file_path):
        with open(cookie_file_path, "r") as f:
            cookie_data = f.read()
            return redirect(url_for("dashboard"))
    else:
        return redirect(url_for("login"))


# login 接口
# @app.route("/qrcode_status", methods=["GET"])
def qrcode_status():
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
                f.write(f"{cookies['SESSDATA']}")
            print(f"SESSDATA 已成功保存到 {cookie_file_path}")
        except Exception as e:
            print(f"保存 SESSDATA 时出错: {e}")
        cookie_data = cookies["SESSDATA"]
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
    return render_template("dashboard.html")


# @app.route("/logout", methods=["POST"])
def logout():
    # 清除保存的 cookies
    if os.path.exists(cookie_file_path):
        os.remove(cookie_file_path)
    return jsonify({"success": True})
