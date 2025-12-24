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

# 新增：导入图像处理库
import io
from PIL import Image

# 关闭matplotlib的警告（可选，避免冗余输出）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100  # 提升绘图分辨率

# 加载训练好的模型（GBD.pkl）
model = joblib.load('GBD.pkl')

# 定义特征名称，对应数据集中的列名
feature_names = [
    "性别", "年龄", "体质指数", "甘油三酯", "低密度脂蛋白胆固醇",
    "高密度脂蛋白胆固醇", "谷丙转氨酶", "谷草酶谷丙酶", "总蛋白", "白蛋白",
    "血肌酐", "血尿酸", "空腹血糖", "白细胞", "淋巴细胞计数",
    "平均血红蛋白", "血小板"
]

# StreamLit 用户界面
st.title("脂肪肝预测器")  # 设置网页标题

# 年龄：数值输入框
年龄 = st.number_input("年龄:", min_value=0, max_value=120, value=41)

# 性别：分类选择框（0：女性，1：男性）
性别 = st.selectbox("性别:", options=[0, 1], format_func=lambda x: "男" if x == 1 else "女")

体质指数 = st.number_input("体质指数:", min_value=0, max_value=30, value=23)
甘油三酯 = st.number_input("甘油三酯:", min_value=0.1, max_value=20.0, value=1.5, step=0.1)
低密度脂蛋白胆固醇 = st.number_input("低密度脂蛋白胆固醇:", min_value=0.5, max_value=10.0, value=2.3, step=0.1)
高密度脂蛋白胆固醇 = st.number_input("高密度脂蛋白胆固醇:", min_value=0.1, max_value=5.0, value=1.2, step=0.1)
谷丙转氨酶 = st.number_input("谷丙转氨酶:", min_value=0, max_value=500, value=20, step=1)
谷草酶谷丙酶 = st.number_input("谷草酶谷丙酶:", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
总蛋白 = st.number_input("总蛋白:", min_value=40, max_value=100, value=70, step=1)
白蛋白 = st.number_input("白蛋白:", min_value=20, max_value=70, value=45, step=1)
血肌酐 = st.number_input("血肌酐:", min_value=10, max_value=500, value=70, step=1)
血尿酸 = st.number_input("血尿酸:", min_value=50, max_value=1000, value=300, step=1)
空腹血糖 = st.number_input("空腹血糖:", min_value=2.0, max_value=20.0, value=5.0, step=0.1)
白细胞 = st.number_input("白细胞:", min_value=1.0, max_value=30.0, value=7.0, step=0.1)
淋巴细胞计数 = st.number_input("淋巴细胞计数:", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
平均血红蛋白 = st.number_input("平均血红蛋白:", min_value=20, max_value=50, value=30, step=1)
血小板 = st.number_input("血小板:", min_value=20, max_value=1000, value=200, step=1)

# 处理输入数据并进行预测
feature_values = [
    性别,年龄,体质指数,甘油三酯,低密度脂蛋白胆固醇,高密度脂蛋白胆固醇,
    谷丙转氨酶,谷草酶谷丙酶,总蛋白,白蛋白,血肌酐,血尿酸,空腹血糖,
    白细胞,淋巴细胞计数,平均血红蛋白,血小板
]  
# 关键：强制转换为float，确保与模型训练时类型一致
feature_values = [float(x) for x in feature_values]
features = np.array([feature_values], dtype=np.float32)
features_df = pd.DataFrame(features, columns=feature_names, dtype=np.float32)

# 当用户点击 "Predict" 按钮时执行以下代码
if st.button("Predict"):
    # 预测类别（0：无脂肪肝，1：有脂肪肝）
    predicted_class = model.predict(features)[0]
    # 预测类别的概率
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**预测类别:** {predicted_class} (1: 有脂肪肝, 0: 无脂肪肝)")
    st.write(f"**预测概率:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"根据模型预测，你有较高的脂肪肝风险。"
            f"模型预测你患脂肪肝的概率为 {probability:.1f}%。"
            "建议及时咨询医生，进行进一步检查和干预。"
        )
    else:
        advice = (
            f"根据模型预测，你患脂肪肝的风险较低。"
            f"模型预测你无脂肪肝的概率为 {probability:.1f}%。"
            "建议保持健康的生活方式，并定期进行体检。"
        )
    st.write(advice)

    # ========== 修正后的SHAP力图绘制 ==========
    st.subheader("预测结果解释（SHAP力图）")
    
    # 清空matplotlib缓存，避免空白
    plt.clf()
    plt.close('all')
    
    # 初始化SHAP解释器
    explainer = shap.TreeExplainer(model)
    # 计算SHAP值（适配分类模型）
    shap_values = explainer.shap_values(features_df)
    
    # 二分类模型：取正类（1）的SHAP值
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    
    # 生成SHAP Force Plot并转为图像
    buf = io.BytesIO()
    # 经典force_plot API（稳定，不易空白）
    shap.force_plot(
        # 基础值（模型的默认预测值）
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        # 当前输入的SHAP值
        shap_values[0],
        # 当前输入的特征值
        features_df.iloc[0],
        # 特征名称
        feature_names=feature_names,
        # 输出名称（可视化时的标签）
        out_names="有脂肪肝概率",
        # 不自动显示
        show=False,
        # 使用matplotlib渲染
        matplotlib=True,
        # 图大小
        figsize=(12, 4)
    )
    # 保存到缓冲区（解决Streamlit渲染空白）
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    # 转为PIL图像
    img = Image.open(buf)
    
    # Streamlit展示图像
    st.image(img, use_column_width=True)
    
    # SHAP力图说明
    st.write("""
    **SHAP力图说明：**
    - 图中红色特征：正向推动预测结果（增加患脂肪肝概率）；
    - 图中蓝色特征：负向推动预测结果（降低患脂肪肝概率）；
    - 特征条的长度：代表该特征对预测结果的影响程度。
    """)

# In[2]:





# In[ ]:




