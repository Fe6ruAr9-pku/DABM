import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import matplotlib.gridspec as gridspec
import time
import pandas as pd

# ==========================================
# 1. 页面配置与自定义 CSS (Nature 风格)
# ==========================================
st.set_page_config(
    page_title="D-ABM Dynamics Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 提高 Matplotlib 的清晰度 (Retina 屏优化)
from IPython.display import set_matplotlib_formats
# 对于 Streamlit，主要是调整 DPI
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 注入自定义 CSS 以实现"Nature"风格排版
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Arial', sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-family: 'Arial', sans-serif;
        font-weight: 700;
        color: #2c3e50;
        font-size: 2.2rem;
    }
    h3 {
        font-family: 'Arial', sans-serif;
        font-weight: 600;
        color: #34495e;
        font-size: 1.2rem;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        height: 3em;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 全局绘图设置 (保持原始风格)
# ==========================================
plt.rcParams['font.family'] = 'Arial'
sns.set(style="whitegrid", context="paper")

# ==========================================
# 3. ABM 核心类 (集成物理移动逻辑)
# ==========================================
class RefinedDisasterABM:
    def __init__(self, N=2000, L=50, risk_ratio=0.3,
                 alpha=0.1, beta=0.5,
                 initial_home_ratio=0.5,
                 base_mobility_rate=0.25,
                 speed=5, 
                 risk_grid_seed=None):
        self.N = N
        self.L = L
        self.alpha = alpha
        self.beta = beta
        self.initial_home_ratio = initial_home_ratio
        self.base_mobility_rate = base_mobility_rate
        self.speed = speed
        self.time_step = 0

        # 环境构建
        if risk_grid_seed is not None:
            np.random.seed(risk_grid_seed)
        self.risk_grid = np.zeros((L, L), dtype=bool)
        num_risk_cells = int(L * L * risk_ratio)
        risk_indices = np.random.choice(L*L, num_risk_cells, replace=False)
        self.risk_grid.ravel()[risk_indices] = True
        
        # 恢复随机种子以免影响Agent生成
        if risk_grid_seed is not None:
            np.random.seed(None)

        self.grid_inflow = np.zeros((L, L))
        self.grid_outflow = np.zeros((L, L))

        # Agent 初始化
        self.agents = []
        for i in range(N):
            loc_home = (np.random.randint(0, L), np.random.randint(0, L))
            loc_work = (np.random.randint(0, L), np.random.randint(0, L))

            if random.random() < initial_home_ratio:
                curr, at_home = loc_home, True
            else:
                curr, at_home = loc_work, False

            self.agents.append({
                'id': i, 'home': loc_home, 'work': loc_work, 'pos': curr,
                'at_home': at_home, 'informed': False, 'evacuating': False,
                'sheltering': False, 'reaction_time': None
            })

    def _move_agent(self, current_pos, target_pos):
        if current_pos == target_pos: return current_pos
        cx, cy = current_pos
        tx, ty = target_pos
        dist_x = tx - cx
        dist_y = ty - cy
        step_x = 0
        if dist_x != 0:
            step_x = int(np.sign(dist_x) * min(abs(dist_x), self.speed))
        step_y = 0
        remaining_speed = self.speed - abs(step_x)
        if dist_y != 0 and remaining_speed > 0:
            step_y = int(np.sign(dist_y) * min(abs(dist_y), remaining_speed))
        return (cx + step_x, cy + step_y)

    def step(self, is_baseline_run=False):
        self.time_step += 1
        active_count = 0
        moves = []

        for agent in self.agents:
            if agent['sheltering']: continue
            will_move = False
            target = None
            in_risk = self.risk_grid[agent['pos']]

            if is_baseline_run:
                if agent['at_home'] and random.random() < self.base_mobility_rate:
                    will_move = True; target = agent['work']
                elif not agent['at_home'] and random.random() < 0.15:
                    will_move = True; target = agent['home']
            else:
                if agent['evacuating']:
                    will_move = True; target = agent['home']
                elif not agent['informed']:
                    if random.random() < self.alpha: agent['informed'] = True

                if agent['informed'] and not agent['evacuating']:
                    prob = self.beta * (1.5 if in_risk else 1.0)
                    if random.random() < prob:
                        if agent['at_home']:
                            agent['sheltering'] = True
                            if agent['reaction_time'] is None: agent['reaction_time'] = self.time_step
                        else:
                            agent['evacuating'] = True
                            will_move = True; target = agent['home']

                if not agent['informed'] and not agent['evacuating']:
                     if agent['at_home'] and random.random() < self.base_mobility_rate:
                        will_move = True; target = agent['work']

            if will_move and target is not None:
                active_count += 1
                old_pos = agent['pos']
                new_pos = self._move_agent(old_pos, target) if not is_baseline_run else target
                moves.append((old_pos, new_pos))
                agent['pos'] = new_pos

                if new_pos == agent['home']:
                    agent['at_home'] = True
                    if agent['evacuating']:
                        agent['sheltering'] = True
                        if agent['reaction_time'] is None: agent['reaction_time'] = self.time_step
                else:
                    agent['at_home'] = False

        for (origin, dest) in moves:
            self.grid_outflow[origin] += 1
            self.grid_inflow[dest] += 1
            
    def get_wrl_data(self):
        data = []
        for a in self.agents:
            if a['reaction_time'] is not None: data.append(a['reaction_time'])
        return data

    def get_agent_positions(self):
        # 优化绘图性能：返回 NumPy 数组
        positions = np.array([a['pos'] for a in self.agents])
        return positions

# ==========================================
# 4. 侧边栏控制
# ==========================================
with st.sidebar:
    # ---------------------------
    # 0. Logo 展示
    # ---------------------------
    try:
        # 尝试加载 Logo，上下排列
        st.image("images/PKU_logo.png", use_container_width=True)
        st.image("images/PKU_logo2.png", use_container_width=True)
    except Exception as e:
        # 仅在调试时打印错误，或者静默失败
        pass

    st.header("⚙️ Simulation Controls")
    
    st.subheader("Environment")
    param_N = st.slider("Population (N)", 500, 5000, 2000, 100)
    param_L = st.slider("Grid Size (L)", 30, 100, 50, 10)
    param_rho = st.slider("Initial Home Ratio (ρ)", 0.0, 1.0, 0.5, 0.1)
    
    st.subheader("Mechanism")
    param_alpha = st.slider("Info Diffusion (α)", 0.0, 1.0, 0.5, 0.05)
    param_beta = st.slider("Compliance (β)", 0.0, 1.0, 0.5, 0.05)
    param_speed = st.slider("Movement Speed", 1, 10, 5)
    
    st.markdown("---")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        # 运行控制区域
        # 1. 连续运行按钮 - 缩短文字以适应按钮宽度
        start_btn = st.button("▶ Run", type="primary", help="Run simulation continuously")
        
        # 2. 单步运行按钮
        step_btn = st.button("⏯ Step", help="Run only one time step.")
        
        # 3. 停止按钮 (使用 session_state 控制循环)
        stop_btn = st.button("⏹ Stop")
        
        # 4. 重置按钮 - 移到左侧与其他按钮对齐，保持布局整洁
        reset_btn = st.button("↺ Reset")

    with col_btn2:
        # 参数调节区域 - 使用更紧凑的标签
        # 1. 每次点击连续运行的步数
        steps_per_click = st.slider("Steps/Run", 1, 200, 50)
        
        # 2. 动画速度 (每帧间隔秒数)
        speed_delay = st.slider("Speed (Delay)", 0.0, 1.0, 0.05, 0.05)
        
        # 这里的重置按钮已经移到左侧了


# ==========================================
# 5. 状态管理
# ==========================================
if 'model' not in st.session_state or reset_btn:
    # 每次重置时，保证 risk_grid 是一样的种子，以便对比 baseline
    seed = 42
    st.session_state.model = RefinedDisasterABM(
        N=param_N, L=param_L, alpha=param_alpha, beta=param_beta, 
        initial_home_ratio=param_rho, speed=param_speed, risk_grid_seed=seed
    )
    # Baseline 模型（用于计算DII/DOI），参数设为无风险反应
    st.session_state.baseline = RefinedDisasterABM(
        N=param_N, L=param_L, alpha=0, beta=0, 
        initial_home_ratio=param_rho, speed=param_speed, risk_grid_seed=seed
    )
    st.session_state.history_wrl = []
    st.session_state.step_count = 0
    st.session_state.is_running = False

# 如果点击了停止按钮，记录状态
if stop_btn:
    st.session_state.stop_requested = True
else:
    st.session_state.stop_requested = False


# ==========================================
# 6. 主逻辑与绘图
# ==========================================

# 标题区
st.title("D-ABM: Spatiotemporal Dynamics Simulator")
st.markdown("Dynamic visualization of Warning Response Latency (WRL) and Flow under different scenarios.")

# 指标看板
st.markdown("### 📊 System Metrics")
cols_metrics = st.columns(4)
metric_placeholders = [col.empty() for col in cols_metrics]

# 新增：象限分布看板
st.markdown("### 🧭 Flow Regime Distribution (Quadrants)")
cols_quad = st.columns(4)
quad_placeholders = [col.empty() for col in cols_quad]

def update_metrics():
    # 1. 更新基础指标
    current_wrl_data = st.session_state.model.get_wrl_data()
    sheltered_count = len(current_wrl_data)
    informed_count = sum(1 for a in st.session_state.model.agents if a['informed'])
    
    metric_placeholders[0].markdown(f"""<div class="metric-card"><div class="metric-value">{st.session_state.step_count}</div><div class="metric-label">Time Step (Hrs)</div></div>""", unsafe_allow_html=True)
    metric_placeholders[1].markdown(f"""<div class="metric-card"><div class="metric-value">{sheltered_count}</div><div class="metric-label">Sheltered Agents</div></div>""", unsafe_allow_html=True)
    metric_placeholders[2].markdown(f"""<div class="metric-card"><div class="metric-value">{informed_count/param_N:.1%}</div><div class="metric-label">Informed Rate</div></div>""", unsafe_allow_html=True)
    metric_placeholders[3].markdown(f"""<div class="metric-card"><div class="metric-value">{len(st.session_state.history_wrl)}</div><div class="metric-label">WRL Samples</div></div>""", unsafe_allow_html=True)

    # 2. 计算并更新象限分布
    # 逻辑与 draw_plots 中一致
    epsilon = 1.0
    risk_mask = st.session_state.model.risk_grid
    
    in_exp = st.session_state.model.grid_inflow[risk_mask]
    out_exp = st.session_state.model.grid_outflow[risk_mask]
    in_base = st.session_state.baseline.grid_inflow[risk_mask]
    out_base = st.session_state.baseline.grid_outflow[risk_mask]
    
    # 默认百分比
    pct_q1, pct_q2, pct_q3, pct_q4 = 0.0, 0.0, 0.0, 0.0
    
    if np.sum(in_base) > 0 or np.sum(out_base) > 0:
        dii = (in_exp + epsilon) / (in_base + epsilon)
        doi = (out_exp + epsilon) / (out_base + epsilon)
        
        total_points = len(dii)
        if total_points > 0:
            # Q1: Transit (High In, High Out)
            c_q1 = np.sum((dii > 1) & (doi > 1))
            # Q2: Source (Low In, High Out)
            c_q2 = np.sum((dii <= 1) & (doi > 1))
            # Q3: Quiet (Low In, Low Out)
            c_q3 = np.sum((dii <= 1) & (doi <= 1))
            # Q4: Stranded (High In, Low Out)
            c_q4 = np.sum((dii > 1) & (doi <= 1))
            
            pct_q1 = c_q1 / total_points
            pct_q2 = c_q2 / total_points
            pct_q3 = c_q3 / total_points
            pct_q4 = c_q4 / total_points

    # 更新象限指标卡片 (使用不同颜色区分)
    quad_placeholders[0].markdown(f"""<div class="metric-card" style="border-left: 5px solid #d57a95;"><div class="metric-value" style="color:#d57a95;">{pct_q1:.1%}</div><div class="metric-label">Transit (Q1)</div></div>""", unsafe_allow_html=True)
    quad_placeholders[1].markdown(f"""<div class="metric-card" style="border-left: 5px solid #5974b8;"><div class="metric-value" style="color:#5974b8;">{pct_q2:.1%}</div><div class="metric-label">Source (Q2)</div></div>""", unsafe_allow_html=True)
    quad_placeholders[2].markdown(f"""<div class="metric-card" style="border-left: 5px solid #BDC3C7;"><div class="metric-value" style="color:#7f8c8d;">{pct_q3:.1%}</div><div class="metric-label">Quiet (Q3)</div></div>""", unsafe_allow_html=True)
    quad_placeholders[3].markdown(f"""<div class="metric-card" style="border-left: 5px solid #F0B27A;"><div class="metric-value" style="color:#F0B27A;">{pct_q4:.1%}</div><div class="metric-label">Stranded (Q4)</div></div>""", unsafe_allow_html=True)


# 初始化显示指标
update_metrics()

# 自定义图例区域 (HTML/CSS)
st.markdown("""
<style>
.legend-container {
    display: flex; 
    flex-wrap: wrap;
    gap: 20px; 
    align-items: center; 
    margin-bottom: 10px; 
    margin-top: 20px;
    padding: 10px 15px; 
    background-color: #ffffff; 
    border-radius: 5px; 
    border: 1px solid #e9ecef;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.legend-item {
    display: flex; 
    align-items: center; 
    gap: 6px;
}
.legend-text {
    font-size: 0.9em; 
    color: #555;
}
</style>
<div class="legend-container">
    <span style="font-weight: bold; color: #2c3e50; margin-right: 5px;">Legend:</span>
    <div class="legend-item">
        <div style="width: 16px; height: 16px; background-color: #fadbd8; border: 1px solid #e6b0aa;"></div>
        <span class="legend-text">Risk Zone</span>
    </div>
    <div class="legend-item">
        <div style="width: 10px; height: 10px; background-color: #2c3e50; border-radius: 50%;"></div>
        <span class="legend-text">Exposed Agent</span>
    </div>
    <div class="legend-item">
        <div style="width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-bottom: 10px solid #e74c3c;"></div>
        <span class="legend-text">Sheltered Agent</span>
    </div>
    <div class="legend-item" style="margin-left: 10px; border-left: 1px solid #ddd; padding-left: 15px;">
        <div style="width: 20px; height: 3px; background-color: #5974b8;"></div>
        <span class="legend-text">Response Density</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 占位符：用于动态更新图表
plot_placeholder = st.empty()

def update_simulation():
    # 同时推演实验组和基准组
    st.session_state.model.step(is_baseline_run=False)
    st.session_state.baseline.step(is_baseline_run=True)
    st.session_state.step_count += 1
    st.session_state.history_wrl = st.session_state.model.get_wrl_data()

def draw_plots():
    # 创建 Nature 风格的组合图
    fig = plt.figure(figsize=(18, 6), constrained_layout=True, dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], figure=fig)

    # ---------------------------
    # Plot 1: Agent Map (Spatial)
    # ---------------------------
    ax1 = fig.add_subplot(gs[0])
    
    # 绘制风险区域底图
    risk_grid = st.session_state.model.risk_grid
    # 使用自定义cmap绘制风险区
    cmap_risk = sns.color_palette(["#f4f6f7", "#fadbd8"], as_cmap=True) # 极淡的红色表示风险
    sns.heatmap(risk_grid.T, ax=ax1, cbar=False, cmap=cmap_risk, alpha=0.6)
    
    # 提取Agent位置
    positions = st.session_state.model.get_agent_positions()
    if len(positions) > 0:
        # 分类绘制：Sheltered vs Exposed
        agents = st.session_state.model.agents
        sheltered_idx = [i for i, a in enumerate(agents) if a['sheltering']]
        exposed_idx = [i for i, a in enumerate(agents) if not a['sheltering']]
        
        # 散点图
        if exposed_idx:
            pos_exp = positions[exposed_idx]
            ax1.scatter(pos_exp[:, 0] + 0.5, pos_exp[:, 1] + 0.5, c='#2c3e50', s=5, alpha=0.4, label='Exposed')
        if sheltered_idx:
            pos_she = positions[sheltered_idx]
            ax1.scatter(pos_she[:, 0] + 0.5, pos_she[:, 1] + 0.5, c='#e74c3c', s=10, alpha=0.8, marker='^', label='Sheltered')
    
    ax1.set_xlim(0, param_L); ax1.set_ylim(0, param_L)
    ax1.set_title("A. Real-time Agent Distribution", fontweight='bold', fontsize=12)
    ax1.axis('off') # 去掉坐标轴使地图更清晰
    # ax1.legend(loc='upper right', frameon=True, fontsize=8) # 移除图内 Legend

    # ---------------------------
    # Plot 2: WRL Distribution
    # ---------------------------
    ax2 = fig.add_subplot(gs[1])
    wrl_data = st.session_state.history_wrl
    
    if len(wrl_data) > 2:
        # 添加抖动以平滑显示
        jittered = np.array(wrl_data) + np.random.uniform(-0.5, 0.5, len(wrl_data))
        sns.histplot(jittered, kde=True, ax=ax2, color="#5974b8", stat="density", binwidth=1, line_kws={'linewidth': 2})
        ax2.set_xlim(0, max(24, st.session_state.step_count))
        # ax2.legend(loc='upper right', frameon=True) # 移除图内 Legend
    else:
        ax2.text(0.5, 0.5, "Waiting for response data...", ha='center', va='center', color='gray')
    
    ax2.set_title("B. Warning Response Latency (WRL)", fontweight='bold', fontsize=12)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Density")

    # ---------------------------
    # Plot 3: DII / DOI Quadrant
    # ---------------------------
    ax3 = fig.add_subplot(gs[2])
    
    # 计算 DII/DOI
    epsilon = 1.0
    risk_mask = st.session_state.model.risk_grid
    
    # 获取累积流
    in_exp = st.session_state.model.grid_inflow[risk_mask]
    out_exp = st.session_state.model.grid_outflow[risk_mask]
    in_base = st.session_state.baseline.grid_inflow[risk_mask]
    out_base = st.session_state.baseline.grid_outflow[risk_mask]
    
    if np.sum(in_base) > 0 or np.sum(out_base) > 0:
        dii = (in_exp + epsilon) / (in_base + epsilon)
        doi = (out_exp + epsilon) / (out_base + epsilon)
        
        # 添加一些随机噪声模拟测量误差，避免完全重叠
        dii *= np.random.uniform(0.98, 1.02, size=len(dii))
        doi *= np.random.uniform(0.98, 1.02, size=len(doi))
        
        # 气泡大小对应活动量
        activity = in_exp + out_exp
        sizes = 20
        if len(activity) > 0 and activity.max() > activity.min():
            sizes = ((activity - activity.min()) / (activity.max() - activity.min() + 1e-6)) * 100 + 20
        
        # 颜色逻辑
        colors = []
        for x, y in zip(dii, doi):
            if x > 1 and y > 1: colors.append('#d57a95') # Q1
            elif x <= 1 and y > 1: colors.append('#5974b8') # Q2
            elif x <= 1 and y <= 1: colors.append('#BDC3C7') # Q3
            else: colors.append('#F0B27A') # Q4
            
        # 绘制背景分区
        limit = 5
        ax3.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax3.axvline(x=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax3.scatter(dii, doi, s=sizes, c=colors, alpha=0.7, edgecolors='white', linewidth=0.5)
        
        # 象限标注
        ax3.text(limit*0.95, limit*0.95, "Transit", ha='right', va='top', fontsize=9, fontweight='bold', color='#d57a95')
        ax3.text(0.05, limit*0.95, "Source", ha='left', va='top', fontsize=9, fontweight='bold', color='#5974b8')
        ax3.text(limit*0.95, 0.05, "Stranded", ha='right', va='bottom', fontsize=9, fontweight='bold', color='#F0B27A')
        
        ax3.set_xlim(0, limit); ax3.set_ylim(0, limit)
    else:
        ax3.text(0.5, 0.5, "Accumulating flow data...", ha='center', va='center', color='gray')
        
    ax3.set_title("C. Flow Regime (DII vs DOI)", fontweight='bold', fontsize=12)
    ax3.set_xlabel("Inflow Index (DII)")
    ax3.set_ylabel("Outflow Index (DOI)")

    return fig

# 始终绘制当前状态，确保交互后图像不消失
fig = draw_plots()
plot_placeholder.pyplot(fig)

# 按钮触发逻辑
# 逻辑1: 连续运行
if start_btn:
    # 循环运行 steps_per_click 步，实现流畅动画效果
    # 创建一个进度条或状态指示
    status_text = st.empty()
    status_text.text("Running simulation...")
    
    for i in range(steps_per_click):
        # 检查是否请求停止 (注意：Streamlit 的按钮点击是瞬间事件，要在循环中检测停止通常需要更复杂的 session_state 管理，
        # 但在这里简单的 stop_btn 点击会触发 rerun，从而打断这个循环，虽然不是最优雅的中断，但有效)
        # 更平滑的方式是每次循环都 check 一下外部状态，但 Streamlit 的单线程模型限制了这一点。
        # 这里我们依靠用户点击 Stop 按钮触发的 Rerun 来自然终止循环。
        
        update_simulation()
        
        # 绘图并显示
        fig = draw_plots()
        plot_placeholder.pyplot(fig)
        
        # 更新指标
        update_metrics()
        
        # 清理内存
        plt.close(fig)
        
        # 速度控制
        time.sleep(speed_delay) 
    
    status_text.text("Run complete.")

# 逻辑2: 单步运行
if step_btn:
    update_simulation()
    fig = draw_plots()
    plot_placeholder.pyplot(fig)
    update_metrics()
    plt.close(fig)


# 底部说明
st.markdown("---")
st.markdown("""
**Model Methodology:**
* **WRL (Warning Response Latency):** Distribution of time elapsed from simulation start to protective action.
* **DII/DOI:** Dynamic Inflow/Outflow Index comparing the current scenario against a baseline (no-disaster) scenario.
* **Quadrants:** Q1 (High In/High Out) = Transit Hub; Q2 (Low In/High Out) = Evacuation Source; Q4 (High In/Low Out) = Shelter/Stranded area.
""")
