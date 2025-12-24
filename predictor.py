# 导入 Streamlit 库，用于构建 Web 应用
import streamlit as st

# 导入 joblib 库，用于加载和保存机器学习模型
import joblib

# 导入 NumPy 库，用于数值计算
import numpy as np

# 导入 Pandas 库，用于数据处理和操作
import pandas as pd

# 导入 SHAP 库，用于解释机器学习模型的预测
import shap

# 导入 Matplotlib 库，用于数据可视化
import matplotlib.pyplot as plt


# 加载训练好的模型（GBD.pkl）
model = joblib.load('GBD.pkl')


# 从 X_test.csv 文件加载测试数据，以便用于 LIME 解释器
X_test = pd.read_csv("lasso_data.csv", encoding='gbk')
X = X_test.iloc[:, 1:]

# 定义特征名称，对应数据集中的列名
feature_names = ["性别",	
                 "年龄",
                 "体质指数",
                 "甘油三酯",
                 "低密度脂蛋白胆固醇",
                 "高密度脂蛋白胆固醇",
                 "谷丙转氨酶",
                 "谷草酶谷丙酶",
                 "总蛋白",
                 "白蛋白",
                 "血肌酐",
                 "血尿酸",
                 "空腹血糖",
                 "白细胞",
                 "淋巴细胞计数",
                 "平均血红蛋白",
                 "血小板"
]


# StreamLit 用户界面
st.title("脂肪肝预测器")  # 设置网页标题

# 年龄：数值输入框
年龄 = st.number_input("年龄:", min_value=0, max_value=120, value=41)

# 性别：分类选择框（0：女性，1：男性）
性别 = st.selectbox("性别:", options=[0, 1], format_func=lambda x: "男" if x == 1 else "女")

体质指数 = st.number_input("体质指数:", min_value=0, max_value=30, value=23)


甘油三酯 = st.number_input("甘油三酯:", min_value=0.1, max_value=20.0, value=1.5, step=0.1)


低密度脂蛋白胆固醇 = st.number_input("低密度脂蛋白胆固醇:", min_value=0.5,
        max_value=10.0,
        value=2.3,
        step=0.1)


高密度脂蛋白胆固醇 = st.number_input("高密度脂蛋白胆固醇:", min_value=0.1,
        max_value=5.0,
        value=1.2,
        step=0.1)


谷丙转氨酶 = st.number_input("谷丙转氨酶:",  min_value=0,
        max_value=500,
        value=20,
        step=1)


谷草酶谷丙酶 = st.number_input("谷草酶谷丙酶:",  min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1)


总蛋白 = st.number_input("总蛋白:", min_value=40,
        max_value=100,
        value=70,
        step=1)


白蛋白 = st.number_input("白蛋白:", min_value=20,
        max_value=70,
        value=45,
        step=1)

血肌酐 = st.number_input("血肌酐:", min_value=10,
        max_value=500,
        value=70,
        step=1)


血尿酸 = st.number_input("血尿酸:", min_value=50,
        max_value=1000,
        value=300,
        step=1)


空腹血糖 = st.number_input("空腹血糖:", min_value=2.0,
        max_value=20.0,
        value=5.0,
        step=0.1)

白细胞 = st.number_input("白细胞:", min_value=1.0,
        max_value=30.0,
        value=7.0,
        step=0.1)

淋巴细胞计数 = st.number_input("淋巴细胞计数:", min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1)

平均血红蛋白 = st.number_input("平均血红蛋白:", min_value=20,
        max_value=50,
        value=30,
        step=1)

血小板 = st.number_input("血小板:", min_value=20,
        max_value=1000,
        value=200,
        step=1)
# 处理输入数据并进行预测
feature_values = [性别,年龄,体质指数,甘油三酯,低密度脂蛋白胆固醇,高密度脂蛋白胆固醇,谷丙转氨酶,谷草酶谷丙酶,总蛋白,白蛋白,血肌酐,
                  血尿酸,空腹血糖,白细胞,淋巴细胞计数,平均血红蛋白,血小板]  # 将用户输入的特征值存入列表
features = np.array([feature_values])  # 将特征转换为 NumPy 数组，适用于模型输入

# 当用户点击 "Predict" 按钮时执行以下代码
if st.button("Predict"):
    # 预测类别（0：无心脏病，1：有心脏病）
    predicted_class = model.predict(features)[0]
    # 预测类别的概率
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")


    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为 1（高风险）
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of heart disease. "
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )
    # 如果预测类别为 0（低风险）
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )
    # 显示建议
    st.write(advice)

 # ========== 新增：生成并展示SHAP Force图（力图） ==========
    st.subheader("预测结果解释（SHAP力图）")
    # 初始化SHAP解释器（针对树模型）
    explainer = shap.TreeExplainer(model)
    # 计算当前输入特征的SHAP值
    shap_values = explainer(features_df)
    
    # 绘制SHAP Force图（matplotlib版本，适配Streamlit）
    fig, ax = plt.subplots(figsize=(12, 4))  # 设置图的大小
    shap.plots.force(
        shap_values[0],  # 取第一条（仅当前输入）的SHAP值
        matplotlib=True,
        show=False,  # 不自动显示，交给Streamlit处理
        feature_names=feature_names  # 关联特征名称
    )
    plt.tight_layout()  # 调整布局避免文字重叠
    st.pyplot(fig)  # Streamlit展示matplotlib图
    
    # 可选：补充说明SHAP力图含义
    st.write("""
    **SHAP力图说明：**
    - 图中红色特征：正向推动预测结果（增加患脂肪肝概率）；
    - 图中蓝色特征：负向推动预测结果（降低患脂肪肝概率）；
    - 特征条的长度：代表该特征对预测结果的影响程度。
    """)

# In[2]:





# In[ ]:




