import pandas as pd
import jieba
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 设置中文字体，防止画图时中文乱码 (Windows系统通常是SimHei)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("1. 正在加载真实数据...")
# 读取文件（确保文件和代码在同一个文件夹）
df = pd.read_csv('real_data.csv') 

# 打印列名确认
print("原始数据的列名:", df.columns.tolist())

# 重命名列：将 'review' 改为 'text'
if 'review' in df.columns:
    df = df.rename(columns={'review': 'text'})

# 数据清洗：去除空值和重复值
df = df.dropna(subset=['text', 'label'])
df = df.drop_duplicates(subset=['text'])

# 如果数据量大于2000条，我们抽取2000条来加速测试（防止跑得太久）
if len(df) > 2000:
    df = df.sample(n=2000, random_state=42)

print(f"清洗后用于训练的数据量: {len(df)} 条")

print("2. 正在进行中文分词...")
def cut_words(text):
    return " ".join(jieba.cut(str(text)))
df['cut_text'] = df['text'].apply(cut_words)

# 划分训练集和测试集 (80%训练, 20%测试)
X_train, X_test, y_train, y_test = train_test_split(df['cut_text'], df['label'], test_size=0.2, random_state=42)

print("3. 提取特征 (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- 算法 A: 朴素贝叶斯 ---
print("4. 训练朴素贝叶斯...")
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)
nb_acc = accuracy_score(y_test, nb_pred)

# --- 算法 B: 逻辑回归 ---
print("5. 训练逻辑回归...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_pred = lr_model.predict(X_test_vec)
lr_acc = accuracy_score(y_test, lr_pred)

print(f"朴素贝叶斯准确率: {nb_acc:.4f}")
print(f"逻辑回归准确率: {lr_acc:.4f}")

# --- 生成模型对比柱状图 ---
plt.figure(figsize=(6, 4))
models = ['朴素贝叶斯', '逻辑回归']
accuracies = [nb_acc, lr_acc]
plt.bar(models, accuracies, color=['#4CAF50', '#2196F3'])
plt.ylim(0, 1.1)
plt.ylabel('准确率 (Accuracy)')
plt.title('模型性能对比 (真实数据)')
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')
plt.savefig('model_comparison.png')
plt.close()
print("已生成模型对比图: model_comparison.png")

# --- 可解释性分析 (提取权重最高的词) ---
feature_names = vectorizer.get_feature_names_out()
coefs = lr_model.coef_[0]
top_positive_idx = coefs.argsort()[-10:][::-1]
top_negative_idx = coefs.argsort()[:10]

top_pos_words = [feature_names[i] for i in top_positive_idx]
top_neg_words = [feature_names[i] for i in top_negative_idx]

with open('top_words.txt', 'w', encoding='utf-8') as f:
    f.write("最能代表【好评】的词: " + "、".join(top_pos_words) + "\n\n")
    f.write("最能代表【差评】的词: " + "、".join(top_neg_words))

# 保存表现更好的模型
if lr_acc >= nb_acc:
    joblib.dump(lr_model, 'best_model.pkl')
    print("最佳模型 (逻辑回归) 已保存！")
else:
    joblib.dump(nb_model, 'best_model.pkl')
    print("最佳模型 (朴素贝叶斯) 已保存！")
    
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("特征提取器已保存！全部流程执行完毕！")