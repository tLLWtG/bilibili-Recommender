from app.getHistoryData import get_history_data
from app.getHotData import get_hot_data
from app.getRecommandData import get_recommand_data

cookie = "your bilibili cookie"
get_recommand_data(cookie, 50)
get_history_data(cookie, 50)
get_hot_data(cookie, 50)
