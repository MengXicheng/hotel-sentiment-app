# app.py
import streamlit as st
import joblib
import jieba

# 设置页面宽屏显示，并设置标题
st.set_page_config(page_title="酒店评价情感分析", page_icon="🏨", layout="wide")

st.title("🏨 酒店消费评价情感分析系统")
st.markdown("本项目基于真实酒店评价数据，使用机器学习算法自动识别顾客评价是**好评**还是**差评**。")

@st.cache_resource
def load_resources():
    model = joblib.load('best_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_resources()
except FileNotFoundError:
    st.error("未找到模型文件，请先在终端运行 `python train_model.py` 训练模型。")
    st.stop()

# 将页面分为左右两列
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📊 项目说明与数据来源")
    st.info(
        """
        **1. 数据来源：**
        *   真实中文酒店消费评价公开数据集 (ChnSentiCorp)。
        *   **清洗步骤：** 去除了空值、重复评论，保证数据质量。
        
        **2. 使用的算法：**
        *   朴素贝叶斯 (MultinomialNB)
        *   逻辑回归 (LogisticRegression)
        
        **3. 算法原理简述：**
        *   **分词：** 使用 Jieba 库将句子切分成词语。
        *   **特征工程：** 使用 TF-IDF 算法，将词语转换为数字权重，评估词语对分类的重要性。
        """
    )
    
    st.subheader("⚖️ 模型准确率对比")
    # 展示训练生成的图表
    try:
        st.image('model_comparison.png', caption='两种算法的准确率对比 (基于真实数据)')
    except FileNotFoundError:
        st.write("暂无对比图，请先运行训练脚本。")
        
    st.success("分析结论：逻辑回归在此数据集上表现略优，因此被部署为最终模型。")

with col2:
    st.header("✍️ 在线情感预测")
    user_input = st.text_area("请输入一段酒店/住宿评价：", height=150, placeholder="例如：这家酒店环境很好，服务热情，早餐也很丰富！")
    
    if st.button("开始分析", type="primary"):
        if user_input.strip() == "":
            st.warning("请输入内容后再点击分析！")
        else:
            cut_input = " ".join(jieba.cut(user_input))
            input_vec = vectorizer.transform([cut_input])
            prediction = model.predict(input_vec)[0]
            proba = model.predict_proba(input_vec)[0]
            
            st.divider()
            if prediction == 1:
                st.success(f"✅ **预测结果：好评 (正面)**")
                st.write(f"模型置信度：{proba[1]*100:.2f}%")
                st.balloons()
            else:
                st.error(f"❌ **预测结果：差评 (负面)**")
                st.write(f"模型置信度：{proba[0]*100:.2f}%")

    st.divider()
    st.header("🔍 模型可解释性分析")
    st.write("为什么模型会这么判断？我们提取了模型权重，看看哪些词最能影响判断：")
    
    try:
        with open('top_words.txt', 'r', encoding='utf-8') as f:
            words_info = f.read()
            st.info(words_info)
    except FileNotFoundError:
        st.write("暂无特征词分析数据。")