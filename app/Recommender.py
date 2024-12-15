"""
Author: wlaten
Date: 2024-12-15 13:56:25
LastEditTime: 2024-12-15 22:04:41
Discription: file content
"""

from app.getHistoryData import get_history_data
from app.getHotData import get_hot_data
from app.getRecommandData import get_recommand_data

from app.model import *


class Recommender:
    # def __init__(self):
    #     pass

    def __init__(self, cookies):
        """
        初始化，仅需要传入cookies
        下面会调用爬取数据然后建立模型
        """
        self.cookies = cookies
        self.video_pool_hot = []
        self.video_pool_recommend = []

        self.processor = FeatureProcessor()
        self.processed_data, self.labels, self.max_tags = load_and_process_data(
            "historyVideo.json", self.processor
        )
        # self.processed_data, self.labels, self.max_tags = load_and_process_data(None, self.processor, cookies)
        self.class_weights_array = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(self.labels), y=self.labels
        )
        self.class_weights_dict = {
            i: weight for i, weight in enumerate(self.class_weights_array)
        }
        print(f"类别权重: {self.class_weights_dict}")

        self.num_tags = len(self.processor.tag2idx)
        self.num_authors = len(self.processor.author2idx)
        self.embedding_dim = 32

        self.model = VideoRecommender(
            self.num_tags, self.num_authors, self.embedding_dim
        )
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)

        num_epochs = 30

        best_loss = float("inf")
        losses = []
        AUCROC = []
        AveragePrecision = []
        PrecisionAtK = []
        RecallAtK = []

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            loss = train_model(
                self.model,
                self.processed_data,
                self.labels,
                self.optimizer,
                self.class_weights_dict,
            )
            losses.append(loss)
            print(f"Epoch {epoch + 1}, Average Loss: {loss}")

            metrics = evaluate_model(self.model, self.processed_data, self.labels)
            print(f"评估指标: {metrics}")

            AUCROC.append(metrics["AUC-ROC"])
            AveragePrecision.append(metrics["Average Precision"])
            PrecisionAtK.append(metrics["Precision@k"])
            RecallAtK.append(metrics["Recall@k"])

            if loss < best_loss:
                best_loss = loss
                print(f"在 epoch {epoch + 1} 保存最佳模型")
                save_model_and_processor(self.model, self.processor, save_dir)

            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(losses) + 1), losses, "b-", label="Training Loss")
            plt.plot(range(1, len(AUCROC) + 1), AUCROC, "r-", label="AUC-ROC")
            plt.plot(
                range(1, len(AveragePrecision) + 1),
                AveragePrecision,
                "g-",
                label="Average Precision",
            )
            plt.plot(
                range(1, len(PrecisionAtK) + 1), PrecisionAtK, "y-", label="Precision@k"
            )
            plt.plot(range(1, len(RecallAtK) + 1), RecallAtK, "c-", label="Recall@k")

            plt.title("Training Metrics")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "training_loss.png"))
            plt.close()

    def behavior_insert(self, video):
        """
        插入用户行为数据，然后模型进行学习
        video视为一个样本对象，要包含如下特征，类似
        {
            "bvid": "BV1Z6zYYwEfy",
            "pic": "http://i1.hdslb.com/bfs/archive/c0620bd3ee503401f014ba4b2c65483a77b15cff.jpg",
            "author": "高端拆解",
            "view": 29204,
            "like": 500,
            "favorite": 290,
            "coin": 290,
            "share": 75,
            "duration": 351,
            "progress": 175.5,
            "tag": [
                "射频电路",
                "高端电子魅魔",
            ],
            "isfaved": 1,
            "isliked": 0,
            "reply": 111
        }
        """
        pass

    def expand_pool(self, pool_type, video_cnt):
        """
        从热门或推荐视频中扩充推荐池
        pool_type: 推荐池类型，hot或者recommend
        """
        if pool_type == "hot":
            data = get_hot_data(self.cookies, video_cnt)
            self.video_pool_hot += data
        elif pool_type == "recommend":
            data = get_recommand_data(self.cookies, video_cnt)
            self.video_pool_recommend += data

    def recommend(self, pool_type, video_cnt=1):
        """
        返回推荐池中最推荐的视频，返回的是一个视频列表
        返回后会pop掉这些视频
        如果视频不够会自己添加
        pool_type: 推荐池类型，hot或者recommend
        """
        pool = self.video_pool_hot if pool_type == "hot" else self.video_pool_recommend

        if video_cnt > len(pool):
            self.expand_pool(pool_type, video_cnt * 3 - len(pool))

        processed_videos = []
        for video in pool:
            valid_tags = [tag for tag in video["tag"] if tag in self.processor.tag2idx]
            if not valid_tags:
                continue
            processed_videos.append({**video, "tag": valid_tags})

        print(f"找到 {len(processed_videos)} 个有效视频")
        if not processed_videos:
            return []

        video_info = []
        all_tags = []
        all_authors = []
        all_quality_scores = []

        for video in processed_videos:
            tags = [self.processor.tag2idx[tag] for tag in video["tag"]]
            tags = tags + [0] * (self.max_tags - len(tags))

            author_idx = (
                self.processor.author2idx[video["author"]]
                if video["author"] in self.processor.author2idx
                else 0
            )

            quality_score = self.processor.calculate_quality_score(
                video["view"], video["like"], video["favorite"]
            )

            all_tags.append(tags)
            all_authors.append(author_idx)
            all_quality_scores.append(quality_score)

            video_info.append(video)
        processed_data = {
            "tags": np.array(all_tags, dtype=np.int32),
            "author": np.array(all_authors, dtype=np.int32),
            "quality_score": np.array(all_quality_scores, dtype=np.float32),
        }
        predictions = (
            self.model(
                [
                    processed_data["tags"],
                    processed_data["author"],
                    processed_data["quality_score"],
                ]
            )
            .numpy()
            .flatten()
        )

        results = []
        for i, pred in enumerate(predictions):
            results.append({**video_info[i], "rating": float(pred)})

        results.sort(key=lambda x: x["rating"], reverse=True)

        top_videos = results[:video_cnt]
        # for video in top_videos:
        #     print(video)

        return top_videos


if __name__ == "__main__":
    cookie_path = os.path.join("user_data", "cookie.txt")
    with open(cookie_path, "r", encoding="utf-8") as file:
        debug_cookies = file.read().strip()

    recommender = Recommender(debug_cookies)
    print(recommender.recommend("hot", 5))
