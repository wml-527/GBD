import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------------- 全局配置（解决字体/画布问题） --------------------------
# 1. 禁用matplotlib后端的GUI渲染（关键：避免Cloud环境画布崩溃）
plt.switch_backend('Agg')
# 2. 简化字体配置（放弃中文字体，改用英文特征名，彻底消除Glyph警告）
feature_names_en = [
    "Gender", "Age", "BMI", "Triglycerides", "LDL-C", "HDL-C", "ALT", "AST/ALT",
    "Total Protein", "Albumin", "Serum Creatinine", "Uric Acid", "Fasting Blood Glucose",
    "White Blood Cell", "Lymphocyte Count", "Mean Hemoglobin", "Platelet"
]
# 中文特征名（仅用于前端显示）
feature_names_cn = [
    "性别", "年龄", "体质指数", "甘油三酯", "低密度脂蛋白胆固醇", "高密度脂蛋白胆固醇",
    "谷丙转氨酶", "谷草酶谷丙酶", "总蛋白", "白蛋白", "血肌酐", "血尿酸", "空腹血糖",
    "白细胞", "淋巴细胞计数", "平均血红蛋白", "血小板"
]

# -------------------------- 加载模型 --------------------------
try:
    model = joblib.load('GBD.pkl')
except Exception as e:
    st.error(f"模型加载失败：{e}")
    st.stop()

# -------------------------- Streamlit 前端界面 --------------------------
st.title("脂肪肝预测器")

# 构建输入组件（中文显示，特征顺序和模型训练一致）
input_values = []
for cn_name in feature_names_cn:
    if cn_name == "性别":
        val = st.selectbox(cn_name, options=[0, 1], format_func=lambda x: "男" if x == 1 else "女")
    elif cn_name == "年龄":
        val = st.number_input(cn_name, min_value=0, max_value=120, value=41)
    elif cn_name == "体质指数":
        val = st.number_input(cn_name, min_value=0.0, max_value=50.0, value=23.0, step=0.1)
    elif cn_name in ["甘油三酯", "低密度脂蛋白胆固醇", "高密度脂蛋白胆固醇", "谷草酶谷丙酶", "空腹血糖", "白细胞", "淋巴细胞计数"]:
        val = st.number_input(cn_name, min_value=0.1, max_value=20.0, value=1.0, step=0.1)
    elif cn_name in ["谷丙转氨酶", "总蛋白", "白蛋白", "血肌酐", "血尿酸", "平均血红蛋白", "血小板"]:
        val = st.number_input(cn_name, min_value=0, max_value=1000, value=50, step=1)
    input_values.append(val)

# -------------------------- 预测逻辑（核心修复） --------------------------
if st.button("预测"):
    # 1. 构造带英文列名的DataFrame（匹配模型训练时的特征名，消除sklearn警告）
    feat_df = pd.DataFrame([input_values], columns=feature_names_en)
    
    # 2. 预测（异常捕获）
    try:
        predicted_class = model.predict(feat_df)[0]
        predicted_proba = model.predict_proba(feat_df)[0]
    except Exception as e:
        st.error(f"预测失败：{e}")
        st.stop()
    
    # 3. 显示预测结果（中文）
    st.subheader("预测结果")
    class_text = "脂肪肝" if predicted_class == 1 else "无脂肪肝"
    st.write(f"**预测类别：** {class_text}")
    st.write(f"**无脂肪肝概率：** {predicted_proba[0]:.2%}")
    st.write(f"**脂肪肝概率：** {predicted_proba[1]:.2%}")
    
    # 4. 生成建议
    if predicted_class == 1:
        advice = f"⚠️ 模型预测你有较高的脂肪肝风险（概率{predicted_proba[1]:.1f}%），建议：\n" \
                 "1. 咨询消化科/肝病科医生；\n" \
                 "2. 控制饮食，减少高油高糖食物摄入；\n" \
                 "3. 每周至少150分钟中等强度运动；\n" \
                 "4. 定期复查肝功能和腹部B超。"
    else:
        advice = f"✅ 模型预测你脂肪肝风险较低（概率{predicted_proba[0]:.1f}%），建议：\n" \
                 "1. 保持健康饮食和规律运动；\n" \
                 "2. 避免长期饮酒和熬夜；\n" \
                 "3. 每年定期体检。"
    st.write(advice)
    
    # -------------------------- SHAP 解释（彻底绕开matplotlib） --------------------------
    st.subheader("特征影响解释（SHAP）")
    try:
        # 初始化TreeExplainer（分类模型）
        explainer = shap.TreeExplainer(model)
        # 计算SHAP值（带列名的DataFrame，避免维度错误）
        shap_values = explainer.shap_values(feat_df)
        
        # 选择对应类别的SHAP值（1=脂肪肝，0=无脂肪肝）
        shap_val = shap_values[predicted_class][0]
        
        # 关键：用SHAP的HTML版力图（无matplotlib依赖，无字体问题）
        import streamlit.components.v1 as components
        shap_html = shap.force_plot(
            base_value=explainer.expected_value[predicted_class],
            shap_values=shap_val,
            features=feat_df.iloc[0],
            feature_names=feature_names_en,  # 用英文特征名，避免字体问题
            show=False,
            matplotlib=False  # 禁用matplotlib渲染
        ).html()
        
        # 显示HTML力图（高度适配）
        components.html(shap_html, height=400)
        
        # 补充：显示特征影响排序（中文）
        st.write("### 特征影响排序")
        shap_importance = pd.DataFrame({
            "特征": feature_names_cn,
            "SHAP值": shap_val,
            "影响程度": np.abs(shap_val)
        }).sort_values("影响程度", ascending=False)
        st.dataframe(shap_importance, use_container_width=True)
        
    except Exception as e:
        st.warning(f"SHAP解释生成失败：{e}")

# -------------------------- 底部提示 --------------------------
st.caption("注：本预测仅为参考，最终诊断请以医生意见为准。")


# In[2]:





# In[ ]:




