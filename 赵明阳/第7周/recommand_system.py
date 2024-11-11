import numpy as np
import pandas as pd

#在这个示例中：
#首先创建了一个模拟的用户 - 物品评分矩阵。
#定义了计算用户之间余弦相似度的函数。
#计算了所有用户之间的相似度矩阵。
#实现了一个推荐函数，为目标用户推荐未评分的物品。

# 模拟用户-物品评分矩阵
data = {
    'User1': [5, 3, 0, 4],
    'User2': [0, 4, 5, 3],
    'User3': [3, 2, 4, 0],
    'User4': [4, 0, 3, 5]
}
df = pd.DataFrame(data, index=['Item1', 'Item2', 'Item3', 'Item4'])

# 计算用户之间的相似度（这里使用余弦相似度）
def cosine_similarity(user1, user2):
    mask1 = df[user1].notnull()
    mask2 = df[user2].notnull()
    common_mask = mask1 & mask2
    common_items = df.index[common_mask]
    num = np.sum(df.loc[common_items, user1] * df.loc[common_items, user2])
    den = np.sqrt(np.sum(df.loc[common_items, user1]**2)) * np.sqrt(np.sum(df.loc[common_items, user2]**2))
    return num / den

# 计算所有用户之间的相似度矩阵
similarity_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
for user1 in df.columns:
    for user2 in df.columns:
        similarity_matrix.loc[user1, user2] = cosine_similarity(user1, user2)

# 为目标用户进行推荐
def recommend(user):
    scores = {}
    for item in df.index:
        if df.loc[item, user] == 0:
            weighted_sum = 0
            similarity_sum = 0
            for other_user in df.columns:
                if other_user!= user and df.loc[item, other_user]!= 0:
                    similarity = similarity_matrix.loc[user, other_user]
                    weighted_sum += similarity * df.loc[item, other_user]
                    similarity_sum += similarity
            if similarity_sum > 0:
                scores[item] = weighted_sum / similarity_sum
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

recommendations = recommend('User1')
print("推荐给 User1 的物品及评分：")
for item, score in recommendations:
    print(f"{item}: {score}")