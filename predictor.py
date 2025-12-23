import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import warnings

# -------------------------- 全局配置 & 警告屏蔽 --------------------------
# 1. 屏蔽所有无关警告（清理日志）
warnings.filterwarnings('ignore')
# 2. 禁用matplotlib GUI后端（Cloud环境必加）
plt.switch_backend('Agg')
# 3. 固定随机种子（保证结果可复现）
np.random.seed(42)

# -------------------------- 特征名映射（核心：必须和模型训练时完全一致！） --------------------------
# ✅ 关键：feature_names_en 必须和训练模型时的特征列名100%匹配（大小写/顺序/拼写）
feature_names_en = [
    "Gender", "Age", "BMI", "Triglycerides", "LDL-C", "HDL-C", "ALT", "AST/ALT",
    "Total Protein", "Albumin", "Serum Creatinine", "Uric Acid", "Fasting Blood Glucose",
    "White Blood Cell", "Lymphocyte Count", "Mean Hemoglobin", "Platelet"
]
# 中文特征名（仅前端显示）
feature_names_cn = [
    "性别", "年龄", "体质指数", "甘油三酯", "低密度脂蛋白胆固醇", "高密度脂蛋白胆固醇",
    "谷丙转氨酶", "谷草酶/谷丙酶", "总蛋白", "白蛋白", "血肌酐", "血尿酸", "空腹血糖",
    "白细胞", "淋巴细胞计数", "平均血红蛋白", "血小板"
]
# 特征输入配置（类型/范围/默认值，提升用户体验）
feature_configs = {
    "性别": {"type": "select", "options": [0, 1], "format": lambda x: "女" if x == 0 else "男", "default": 1},
    "年龄": {"type": "number", "min": 0, "max": 120, "default": 41, "step": 1},
    "体质指数": {"type": "number", "min": 10.0, "max": 50.0, "default": 23.0, "step": 0.1},
    "甘油三酯": {"type": "number", "min": 0.1, "max": 20.0, "default": 1.5, "step": 0.1},
    "低密度脂蛋白胆固醇": {"type": "number", "min": 0.1, "max": 10.0, "default": 2.8, "step": 0.1},
    "高密度脂蛋白胆固醇": {"type": "number", "min": 0.1, "max": 5.0, "default": 1.2, "step": 0.1},
    "谷丙转氨酶": {"type": "number", "min": 0, "max": 500, "default": 30, "step": 1},
    "谷草酶/谷丙酶": {"type": "number", "min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1},
    "总蛋白": {"type": "number", "min": 0, "max": 100, "default": 70, "step": 1},
    "白蛋白": {"type": "number", "min": 0, "max": 60, "default": 40, "step": 1},
    "血肌酐": {"type": "number", "min": 0, "max": 500, "default": 80, "step": 1},
    "血尿酸": {"type": "number", "min": 0, "max": 1000, "default": 350, "step": 1},
    "空腹血糖": {"type": "number", "min": 2.0, "max": 20.0, "default": 5.5, "step": 0.1},
    "白细胞": {"type": "number", "min": 1.0, "max": 30.0, "default": 6.5, "step": 0.1},
    "淋巴细胞计数": {"type": "number", "min": 0.1, "max": 10.0, "default": 2.5, "step": 0.1},
    "平均血红蛋白": {"type": "number", "min": 10, "max": 50, "default": 28, "step": 1},
    "血小板": {"type": "number", "min": 0, "max": 1000, "default": 200, "step": 1}
}

# -------------------------- 模型加载（增强容错） --------------------------
@st.cache_resource  # 缓存模型，避免重复加载
def load_model():
    try:
        # 优先加载本地模型，Cloud部署时确保GBD.pkl在根目录
        model = joblib.load('GBD.pkl')
        st.success("✅ 模型加载成功")
        return model
    except FileNotFoundError:
        st.error("❌ 模型文件GBD.pkl未找到！请确认文件路径是否正确")
        st.stop()
    except Exception as e:
        st.error(f"❌ 模型加载失败：{str(e)}")
        st.stop()

model = load_model()

# -------------------------- Streamlit 前端界面 --------------------------
st.set_page_config(page_title="脂肪肝预测器", page_icon="🩺", layout="wide")  # 页面配置
st.title("🩺 脂肪肝风险预测器")
st.divider()

# 分栏布局（提升美观度）
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📋 输入检测指标")
    # 构建输入组件（按配置自动生成）
    input_values = []
    for cn_name in feature_names_cn:
        config = feature_configs[cn_name]
        if config["type"] == "select":
            val = st.selectbox(
                cn_name,
                options=config["options"],
                format_func=config["format"],
                index=config["default"],
                key=cn_name  # 唯一key，避免组件冲突
            )
        elif config["type"] == "number":
            val = st.number_input(
                cn_name,
                min_value=config["min"],
                max_value=config["max"],
                value=config["default"],
                step=config["step"],
                key=cn_name
            )
        input_values.append(val)

with col2:
    st.subheader("🎯 预测结果")
    predict_btn = st.button("开始预测", type="primary", use_container_width=True)
    
    if predict_btn:
        # 1. 构造特征DataFrame（核心：英文列名匹配模型）
        feat_df = pd.DataFrame([input_values], columns=feature_names_en)
        
        # 2. 模型预测（增强异常捕获）
        try:
            predicted_class = model.predict(feat_df)[0]
            predicted_proba = model.predict_proba(feat_df)[0]
        except Exception as e:
            st.error(f"预测失败：{str(e)}")
            st.info("💡 可能原因：输入特征数量/顺序与模型训练时不一致")
            st.stop()
        
        # 3. 显示预测结果（可视化+中文）
        class_text = "✅ 无脂肪肝" if predicted_class == 0 else "⚠️ 脂肪肝"
        proba_text = f"{predicted_proba[predicted_class]:.2%}"
        st.metric(label="预测结果", value=class_text, help="模型基于梯度提升树算法预测")
        st.write(f"无脂肪肝概率：{predicted_proba[0]:.2%}")
        st.write(f"脂肪肝概率：{predicted_proba[1]:.2%}")
        
        # 4. 个性化建议
        st.subheader("💡 健康建议")
        if predicted_class == 1:
            advice = f"""
            你有较高的脂肪肝风险（概率{predicted_proba[1]:.1f}%），建议：
            1. 🥗 控制饮食：减少高油、高糖、高盐食物，增加膳食纤维摄入；
            2. 🏃 规律运动：每周至少150分钟中等强度有氧运动（快走/慢跑/游泳）；
            3. 🚫 戒烟限酒：避免长期饮酒，减少肝脏负担；
            4. 🏥 定期复查：建议每6个月检查肝功能和腹部B超。
            """
        else:
            advice = f"""
            你脂肪肝风险较低（概率{predicted_proba[0]:.1f}%），建议：
            1. 🥙 保持健康饮食：继续维持低脂、低糖的饮食习惯；
            2. 🧘 规律作息：避免熬夜，保证7-8小时睡眠；
            3. 📅 年度体检：每年定期做肝功能和腹部B超检查。
            """
        st.write(advice)

# -------------------------- SHAP 解释模块（优化HTML渲染） --------------------------
st.divider()
st.subheader("🔍 特征影响分析（SHAP）")
try:
    # 初始化解释器（分类模型）
    explainer = shap.TreeExplainer(model)
    # 计算SHAP值（用带列名的DataFrame，避免维度错误）
    feat_df = pd.DataFrame([input_values], columns=feature_names_en)
    shap_values = explainer.shap_values(feat_df)
    
    # 处理分类模型SHAP值（二分类返回list，取对应类别的值）
    if isinstance(shap_values, list):
        # 显示"脂肪肝"类别的特征影响（更贴合用户关注）
        shap_val = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        shap_val = shap_values[0]
        base_value = explainer.expected_value
    
    # 生成SHAP HTML力图（核心优化：解决渲染问题）
    shap_force_plot = shap.force_plot(
        base_value=base_value,
        shap_values=shap_val,
        features=feat_df.iloc[0],
        feature_names=feature_names_en,
        show=False,
        matplotlib=False,
        text_rotation=0,
        plot_cmap=["#FF9999", "#66B2FF"]  # 自定义配色（红=负向，蓝=正向）
    )
    
    # 转换HTML并显示（修复Streamlit渲染高度问题）
    import streamlit.components.v1 as components
    shap_html = f"""
    <div style="width:100%; overflow-x:auto;">
        {shap_force_plot.html()}
    </div>
    """
    components.html(shap_html, height=200, scrolling=True)
    
    # 补充特征影响排序（中文显示，更易理解）
    st.subheader("📊 特征影响排序（中文）")
    shap_importance = pd.DataFrame({
        "特征名称": feature_names_cn,
        "SHAP值（影响程度）": shap_val,
        "绝对影响": np.abs(shap_val)
    }).sort_values("绝对影响", ascending=False)
    # 高亮TOP5特征
    def highlight_top5(row):
        return ['background-color: #f0f8ff' if row.name < 5 else '' for _ in row]
    st.dataframe(
        shap_importance.style.apply(highlight_top5, axis=1),
        use_container_width=True,
        hide_index=True
    )

except Exception as e:
    st.warning(f"SHAP分析暂时不可用：{str(e)}")
    st.info("💡 可能原因：模型类型不支持TreeExplainer / 特征维度不匹配")

# -------------------------- 底部说明 --------------------------
st.divider()
st.caption("⚠️ 免责声明：本工具仅为健康风险参考，不构成医疗诊断，最终请以专业医生意见为准。")


# In[2]:





# In[ ]:




