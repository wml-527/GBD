#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
# 新增：解决图片保存的内存IO问题
from io import BytesIO
# 新增：设置中文字体（适配Streamlit Cloud）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# -------------------------- 加载模型和特征配置 --------------------------
# 加载训练好的模型（GBD.pkl）
model = joblib.load('GBD.pkl')

# 定义特征名称（必须和模型训练时的特征顺序/名称完全一致！）
feature_names = [
    "性别", "年龄", "体质指数", "甘油三酯", "低密度脂蛋白胆固醇",
    "高密度脂蛋白胆固醇", "谷丙转氨酶", "谷草酶谷丙酶", "总蛋白", "白蛋白",
    "血肌酐", "血尿酸", "空腹血糖", "白细胞", "淋巴细胞计数",
    "平均血红蛋白", "血小板"
]

# -------------------------- StreamLit 用户界面 --------------------------
st.title("脂肪肝预测器")  # 设置网页标题

# 年龄：数值输入框
年龄 = st.number_input("年龄:", min_value=0, max_value=120, value=41)
# 性别：分类选择框（0：女性，1：男性）
性别 = st.selectbox("性别:", options=[0, 1], format_func=lambda x: "男" if x == 1 else "女")
体质指数 = st.number_input("体质指数:", min_value=0, max_value=30, value=23)
甘油三酯 = st.number_input("甘油三酯:", min_value=0.1, max_value=20.0, value=1.5, step=0.1)
低密度脂蛋白胆固醇 = st.number_input("低密度脂蛋白胆固醇:", min_value=0.5, max_value=10.0, value=2.3, step=0.1)
高密度脂蛋白胆固醇 = st.number_input("高密度脂蛋白胆固醇:", min_value=0.1, max_value=5.0, value=1.2, step=0.1)
谷丙转氨酶 = st.number_input("谷丙转氨酶:",  min_value=0, max_value=500, value=20, step=1)
谷草酶谷丙酶 = st.number_input("谷草酶谷丙酶:",  min_value=0.1, max_value=5.0, value=1.0, step=0.1)
总蛋白 = st.number_input("总蛋白:", min_value=40, max_value=100, value=70, step=1)
白蛋白 = st.number_input("白蛋白:", min_value=20, max_value=70, value=45, step=1)
血肌酐 = st.number_input("血肌酐:", min_value=10, max_value=500, value=70, step=1)
血尿酸 = st.number_input("血尿酸:", min_value=50, max_value=1000, value=300, step=1)
空腹血糖 = st.number_input("空腹血糖:", min_value=2.0, max_value=20.0, value=5.0, step=0.1)
白细胞 = st.number_input("白细胞:", min_value=1.0, max_value=30.0, value=7.0, step=0.1)
淋巴细胞计数 = st.number_input("淋巴细胞计数:", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
平均血红蛋白 = st.number_input("平均血红蛋白:", min_value=20, max_value=50, value=30, step=1)
血小板 = st.number_input("血小板:", min_value=20, max_value=1000, value=200, step=1)

# -------------------------- 预测逻辑（修复核心问题） --------------------------
# 构造特征数据（带列名，解决sklearn警告）
feature_values = [
    性别,年龄,体质指数,甘油三酯,低密度脂蛋白胆固醇,
    高密度脂蛋白胆固醇,谷丙转氨酶,谷草酶谷丙酶,总蛋白,白蛋白,
    血肌酐,血尿酸,空腹血糖,白细胞,淋巴细胞计数,
    平均血红蛋白,血小板
]
# 关键：用DataFrame封装特征（带列名，匹配模型训练时的特征名）
feat_df = pd.DataFrame([feature_values], columns=feature_names)

# 当用户点击 "Predict" 按钮时执行以下代码
if st.button("Predict"):
    # 修复1：用带列名的DataFrame预测，消除特征名警告
    predicted_class = model.predict(feat_df)[0]
    predicted_proba = model.predict_proba(feat_df)[0]

    # 修复2：修正术语（心脏病→脂肪肝）
    st.write(f"**预测结果:** {predicted_class} (1: 脂肪肝, 0: 无脂肪肝)")
    st.write(f"**预测概率:** 无脂肪肝 {predicted_proba[0]:.2%} | 脂肪肝 {predicted_proba[1]:.2%}")

    # 生成建议（修正术语）
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"根据模型预测，你有较高的脂肪肝风险，预测概率为 {probability:.1f}%。"
            "建议及时咨询消化科/肝病科医生，调整饮食结构、增加运动，并定期复查肝功能和腹部B超。"
        )
    else:
        advice = (
            f"根据模型预测，你当前脂肪肝风险较低，预测概率为 {probability:.1f}%。"
            "建议保持健康的饮食和运动习惯，定期进行体检，预防脂肪肝发生。"
        )
    st.write(advice)

    # -------------------------- SHAP 解释（彻底修复绘图崩溃） --------------------------
    st.subheader("SHAP 特征影响解释")
    # 初始化SHAP解释器
    explainer_shap = shap.TreeExplainer(model)
    # 计算SHAP值（用带列名的feat_df，避免维度错误）
    shap_values = explainer_shap.shap_values(feat_df)

    # 修复3：改用SHAP HTML版力图（绕开matplotlib崩溃）
    import streamlit.components.v1 as components
    # 选择对应类别的SHAP值（分类模型返回两个类别的SHAP值）
    shap_val = shap_values[predicted_class][0]
    # 生成HTML格式的力图（无matplotlib依赖）
    shap_html = shap.force_plot(
        base_value=explainer_shap.expected_value[predicted_class],
        shap_values=shap_val,
        features=feat_df.iloc[0],
        feature_names=feature_names,
        matplotlib=False,  # 禁用matplotlib，避免崩溃
        show=False
    ).html()
    # 在Streamlit中显示HTML力图
    components.html(shap_html, height=300)

    # （可选）保留matplotlib版力图（修复dpi和字体问题）
    # 若仍想显示图片版，用以下代码替换上述HTML部分：
    """
    st.subheader("SHAP 特征影响图（图片版）")
    # 初始化画布，避免NoneType错误
    fig, ax = plt.subplots(figsize=(10, 6))
    # 绘制SHAP力图
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[1], feat_df, ax=ax, show=False)
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[0], feat_df, ax=ax, show=False)
    # 修复4：内存中保存图片（避免Cloud路径权限问题）
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    # 显示图片
    st.image(buf, caption='SHAP Force Plot Explanation')
    # 清理画布，避免内存泄漏
    plt.close(fig)
    """


# In[2]:





# In[ ]:




