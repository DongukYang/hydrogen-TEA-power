# app.py (전체 수정 반영 버전)

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import numpy_financial as nf

# ---------------- 기본 설정 ----------------
st.set_page_config(page_title="수소생산 경제성분석 결과", layout="wide")
st.title("수소생산 경제성분석 결과 [Ver.1]")

# ---------------- 1. 입력값 (사이드바) ----------------
st.sidebar.header("입력값 (User Input)")

# 1-1. 기술성 입력
st.sidebar.subheader("기술성")

# re_capacity_kw = st.sidebar.number_input(
#     "재생에너지 설비용량 (kW)", min_value=0.0, value=10000.0, step=100.0
# )
h2_capacity_kw = st.sidebar.number_input(
    "수소생산 설비용량 (kW)", min_value=0.0, value=5000.0, step=100.0
)
construction_years = st.sidebar.number_input(
    "건설기간 (년)", min_value=0.0, value=2.0, step=0.5
)
operation_years = st.sidebar.number_input(
    "운영기간 (년)", min_value=1.0, value=20.0, step=1.0
)
stack_replacement_hours = st.sidebar.number_input(
    "스택 교체주기 (시간)", min_value=0.0, value=60000.0, step=1000.0
)
annual_operating_hours = st.sidebar.number_input(
    "연간 운전시간 (시간/년)", min_value=0.0, max_value=8760.0, value=8000.0, step=100.0
)
specific_energy = st.sidebar.number_input(
    "에너지 효율 (kWh/kgH₂)", min_value=1.0, value=55.5, step=0.1
)
stack_degradation = st.sidebar.number_input(
    "스택 효율 감소율 (%/년)", min_value=0.0, max_value=10.0, value=1.0, step=0.1
)

# 1-2. 경제성 입력
st.sidebar.subheader("경제성")

# 수소설비 500 kW 기준
capex_construction = st.sidebar.number_input(
    "CAPEX-공사비 (원)",
    min_value=0.0,
    value=100_000_000.0 * (h2_capacity_kw / 500.0),
    step=50_000_000.0,
)
capex_equipment = st.sidebar.number_input(
    "CAPEX-설비비 (원)",
    min_value=0.0,
    value=1_000_000_000.0 * (h2_capacity_kw / 500.0),
    step=100_000_000.0,
)

# 합계 CAPEX (계산용 변수)
capex_total = capex_construction + capex_equipment

opex_annual = st.sidebar.number_input(
    "OPEX (연간, 원/년)", min_value=0.0, value=3_000_000_000.0, step=100_000_000.0
)
elec_price = st.sidebar.number_input(
    "전기요금 (원/kWh)", min_value=0.0, value=50.0, step=1.0
)
h2_price = st.sidebar.number_input(  
    "수소 판매가격 (원/kgH₂)", min_value=0.0, value=10000.0, step=100.0
)
o2_price = st.sidebar.number_input(
    "산소 판매가격 (원/kgO₂)", min_value=0.0, value=50.0, step=1.0
)
heat_price = st.sidebar.number_input(
    "열 판매가격 (원/MWh)", min_value=0.0, value=0.0, step=10.0
)
discount_rate = st.sidebar.number_input(
    "할인율 (%/년)", min_value=0.0, max_value=20.0, value=7.0, step=0.1
)
inflation_rate = st.sidebar.number_input(
    "물가상승률 (%/년)", min_value=0.0, max_value=10.0, value=2.0, step=0.1
)
cost_of_capital = st.sidebar.number_input(
    "총자본비용 (%/년)", min_value=0.0, max_value=20.0, value=8.0, step=0.1
)
corp_tax_rate = st.sidebar.number_input(
    "법인세율 (%):",
    min_value=0.0,
    max_value=50.0,
    value=20.0,
    step=0.1
)
tax_rate = corp_tax_rate / 100.0

# ---------------- 2. LCOH · 재무 모델 ----------------

# 연간 수소 생산량 (kgH2/년) : P(kW) * 시간(h) / (kWh/kgH2)
annual_h2_kg = 0.0
if specific_energy > 0:
    annual_h2_kg = (h2_capacity_kw * annual_operating_hours) / specific_energy

# ===== 스택 효율 저하 반영 =====
degradation_rate = stack_degradation / 100.0  # % → 소수
stack_factor_year = [(1 - degradation_rate) ** t for t in range(int(operation_years))]

# 연간 전력 사용량 및 전력비용
annual_elec_kwh = h2_capacity_kw * annual_operating_hours
annual_elec_cost = annual_elec_kwh * elec_price

# 부제품 산소 및 열 연간 생산/수익
annual_o2_kg = annual_h2_kg * 8.0  # 물 전기분해: H2 1kg당 O2 약 8kg 생성
annual_o2_revenue = annual_o2_kg * o2_price
heat_mwh_per_kg = 0.0  # (필요시 실제 데이터로 수정)
annual_heat_mwh = annual_h2_kg * heat_mwh_per_kg
annual_heat_revenue = annual_heat_mwh * heat_price

# 실질 할인율 계산 (피셔 공식 적용)
r_nom = discount_rate / 100.0
inf = inflation_rate / 100.0
if (1 + inf) > 0:
    r_real = (1 + r_nom) / (1 + inf) - 1
else:
    r_real = r_nom

# NPV/BEP 계산을 위한 현금흐름 생성
cash_flows = [-capex_total]  # t=0 초기 투자
pv_in = 0.0
pv_out = capex_total  # 초기 CAPEX 현재가치(=자본 유출)

# 스택 열화 및 교체 이력 변수 초기화
stack_deg_rate = stack_degradation / 100.0
hours_since_last_replacement = 0.0
years_since_last_replacement = 0

for t in range(1, int(operation_years) + 1):
    # 스택 교체 여부 검사
    opex_stack_t = 0.0
    if stack_replacement_hours > 0 and hours_since_last_replacement >= stack_replacement_hours:
        opex_stack_t = opex_annual * 0.3  # (교체비용 가정: 연간 OPEX의 30%)
        hours_since_last_replacement = 0.0
        years_since_last_replacement = 0

    # 효율 열화 반영한 생산량
    degradation_multiplier = (1 - stack_deg_rate) ** years_since_last_replacement if stack_deg_rate > 0 else 1.0
    h2_t = annual_h2_kg * degradation_multiplier
    elec_kwh_t = annual_elec_kwh * degradation_multiplier

    # 연간 비용 및 수익
    yearly_cost = elec_kwh_t * elec_price + opex_annual + opex_stack_t  # 전력비+고정/가변OPEX+교체비
    yearly_rev = (h2_t * h2_price) + (h2_t * 8.0 * o2_price) + (h2_t * heat_mwh_per_kg * heat_price)

    # 현금흐름 및 현재가치 합산
    net_cf = yearly_rev - yearly_cost
    cash_flows.append(net_cf)
    if r_real > -0.9999:
        pv_in += yearly_rev / ((1 + r_real) ** t)
        pv_out += yearly_cost / ((1 + r_real) ** t)

    hours_since_last_replacement += annual_operating_hours
    years_since_last_replacement += 1

npv = pv_in - pv_out
bc_ratio = pv_in / pv_out if pv_out > 0 else np.nan

try:
    irr = nf.irr(cash_flows)
    p_irr_pct = irr * 100 if irr is not None else np.nan
except Exception:
    p_irr_pct = np.nan

# LCOH 계산
depreciation_amount = capex_total / operation_years if operation_years > 0 else 0.0
numerator_sum_after_tax = capex_total
denominator_sum = 0.0
# 운영 첫 해부터 마지막 해까지 연도별 순비용 계산
hours_since_last_replacement = 0.0
years_since_last_replacement = 0
for t in range(1, int(operation_years) + 1):
    opex_stack_t = 0.0
    if stack_replacement_hours > 0 and hours_since_last_replacement >= stack_replacement_hours:
        opex_stack_t = opex_annual * 0.3
        hours_since_last_replacement = 0.0
        years_since_last_replacement = 0
    degradation_multiplier = (1 - stack_deg_rate) ** years_since_last_replacement if stack_deg_rate > 0 else 1.0
    h2_t = annual_h2_kg * degradation_multiplier
    elec_kwh_t = annual_elec_kwh * degradation_multiplier
    cost_t = elec_kwh_t * elec_price + opex_annual + opex_stack_t
    o2_rev_t = h2_t * 8.0 * o2_price
    heat_rev_t = h2_t * heat_mwh_per_kg * heat_price
    numerator_sum_after_tax += ((cost_t - o2_rev_t - heat_rev_t) * (1 - tax_rate) + tax_rate * depreciation_amount) / ((1 + r_real) ** t)
    denominator_sum += h2_t / ((1 + r_real) ** t)
    hours_since_last_replacement += annual_operating_hours
    years_since_last_replacement += 1

if denominator_sum > 0:
    lcoh_krw_per_kg = numerator_sum_after_tax / denominator_sum / (1 - tax_rate)
else:
    lcoh_krw_per_kg = 0.0

# LCOH 계산용 함수 (민감도 분석에 활용)
def compute_lcoh_given_params(capex_total_val, specific_energy_val, elec_price_val, annual_operating_hours_val, opex_annual_val, discount_rate_val):
    # 실질 할인율 계산 (물가상승률 inf는 고정)
    r_nominal = discount_rate_val / 100.0
    r_real_val = (1 + r_nominal) / (1 + inf) - 1 if (1 + inf) > 0 else r_nominal
    # 시나리오 연간 수소 생산량 및 전력사용량
    annual_h2_val = 0.0
    if specific_energy_val > 0:
        annual_h2_val = (h2_capacity_kw * annual_operating_hours_val) / specific_energy_val
    annual_elec_kwh_val = h2_capacity_kw * annual_operating_hours_val
    # 현금흐름 합산 변수 초기화
    hours_since_last_replacement_val = 0.0
    years_since_last_replacement_val = 0
    numerator_val = capex_total_val
    denominator_val = 0.0
    # 감가상각 (정액법)
    depreciation_val = capex_total_val / operation_years if operation_years > 0 else 0.0
    for t in range(1, int(operation_years) + 1):
        opex_stack_repl = 0.0
        if stack_replacement_hours > 0 and hours_since_last_replacement_val >= stack_replacement_hours:
            opex_stack_repl = opex_annual_val * 0.3
            hours_since_last_replacement_val = 0.0
            years_since_last_replacement_val = 0
        deg_multiplier = (1 - stack_degradation / 100.0) ** years_since_last_replacement_val if stack_degradation > 0 else 1.0
        h2_t_val = annual_h2_val * deg_multiplier
        elec_kwh_t_val = annual_elec_kwh_val * deg_multiplier
        cost_t_val = elec_kwh_t_val * elec_price_val + opex_annual_val + opex_stack_repl
        o2_rev_t_val = h2_t_val * 8.0 * o2_price
        heat_rev_t_val = h2_t_val * heat_mwh_per_kg * heat_price
        numerator_val += ((cost_t_val - o2_rev_t_val - heat_rev_t_val) * (1 - tax_rate) + tax_rate * depreciation_val) / ((1 + r_real_val) ** t)
        denominator_val += h2_t_val / ((1 + r_real_val) ** t)
        hours_since_last_replacement_val += annual_operating_hours_val
        years_since_last_replacement_val += 1
    if denominator_val > 0:
        return numerator_val / denominator_val / (1 - tax_rate)
    else:
        return 0.0

# --- LCOH 구성요소 데이터 (원/kgH2 단위) ---
labels = ["CAPEX-공사", "CAPEX-설비", "OPEX-O&M", "OPEX-전력", "Total LCOH"]

# 초기값
numerator_construction = capex_construction   
numerator_equipment   = capex_equipment       
numerator_opex        = 0.0                  
numerator_elec        = 0.0                  
numerator_byprod      = 0.0                  

# 할인된 비용/수익
hours_since_last_replacement_val = 0.0
years_since_last_replacement_val = 0
for t in range(1, int(operation_years) + 1):
    # 스택교체
    opex_stack_t = 0.0
    if stack_replacement_hours > 0 and hours_since_last_replacement_val >= stack_replacement_hours:
        opex_stack_t = opex_annual * 0.3  # 스택교체비용 OPEX의 30% 가정
        hours_since_last_replacement_val = 0.0
        years_since_last_replacement_val = 0

    # 스택 효율
    deg_multiplier = (1 - stack_deg_rate) ** years_since_last_replacement_val if stack_deg_rate > 0 else 1.0

    # 수소생산량, 전력사용량
    h2_t = annual_h2_kg * deg_multiplier
    elec_kwh_t = annual_elec_kwh * deg_multiplier

    # 연간 비용 수익
    elec_cost_t = elec_kwh_t * elec_price               # 전력비용
    opex_cost_t = opex_annual + opex_stack_t            # O&M cost (fixed OPEX + 교체비용)
    o2_rev_t   = h2_t * 8.0 * o2_price                  # 산소 판매수익 (8 kg O₂ per 1 kg H₂)
    heat_rev_t = h2_t * heat_mwh_per_kg * heat_price    # 열 판매수익

    # 할인율
    discount_factor = 1 / ((1 + r_real) ** t) if r_real != -1 else 1.0 

    # Accumulate present value of each component (after-tax)
    numerator_elec += elec_cost_t * (1 - tax_rate) * discount_factor        # 세후 전력비
    numerator_opex += opex_cost_t * (1 - tax_rate) * discount_factor        # 세후 O&M
    numerator_byprod += - (o2_rev_t + heat_rev_t) * (1 - tax_rate) * discount_factor  # 수익

    numerator_construction += tax_rate * (depreciation_amount * (capex_construction / capex_total)) * discount_factor
    numerator_equipment   += tax_rate * (depreciation_amount * (capex_equipment   / capex_total)) * discount_factor

    hours_since_last_replacement_val += annual_operating_hours
    years_since_last_replacement_val += 1

capex_construction_per_kg = numerator_construction / denominator_sum / (1 - tax_rate) if denominator_sum > 0 else 0.0
capex_equipment_per_kg   = numerator_equipment   / denominator_sum / (1 - tax_rate) if denominator_sum > 0 else 0.0
opex_per_kg              = (numerator_opex + numerator_byprod) / denominator_sum / (1 - tax_rate) if denominator_sum > 0 else 0.0
elec_per_kg              = numerator_elec        / denominator_sum / (1 - tax_rate) if denominator_sum > 0 else 0.0

# total LCOH
total_lcoh_calc = capex_construction_per_kg + capex_equipment_per_kg + opex_per_kg + elec_per_kg

# 파이차트/비용비율 산출용 (Total LCOH 제외)
drivers = ["CAPEX-공사", "CAPEX-설비", "OPEX-O&M", "OPEX-전력"]
values  = [capex_construction_per_kg, capex_equipment_per_kg, opex_per_kg, elec_per_kg]

# 색상
colors = {
    "CAPEX-공사": "#1f77b4",
    "CAPEX-설비": "#aec7e8",
    "OPEX-O&M":  "#d62728",
    "OPEX-전력": "#ff9896",
}

# Bar Chart
y_capex_constr = [capex_construction_per_kg, 0, 0, 0, capex_construction_per_kg]
y_capex_equip  = [0, capex_equipment_per_kg, 0, 0, capex_equipment_per_kg]
y_opex         = [0, 0, opex_per_kg, 0, opex_per_kg]
y_elec         = [0, 0, 0, elec_per_kg, elec_per_kg]

fig_bar = go.Figure()
fig_bar.add_bar(name="CAPEX-공사", x=labels, y=y_capex_constr,
               text=[f"{v:,.0f}" if v > 0 else "" for v in y_capex_constr], textposition="auto", marker_color=colors["CAPEX-공사"])
fig_bar.add_bar(name="CAPEX-설비", x=labels, y=y_capex_equip,
               text=[f"{v:,.0f}" if v > 0 else "" for v in y_capex_equip], textposition="auto", marker_color=colors["CAPEX-설비"])
fig_bar.add_bar(name="OPEX-O&M", x=labels, y=y_opex,
               text=[f"{v:,.0f}" if v > 0 else "" for v in y_opex], textposition="auto", marker_color=colors["OPEX-O&M"])
fig_bar.add_bar(name="OPEX-전력", x=labels, y=y_elec,
               text=[f"{v:,.0f}" if v > 0 else "" for v in y_elec], textposition="auto", marker_color=colors["OPEX-전력"])

fig_bar.update_layout(barmode="stack", yaxis_title="원/kgH₂", xaxis_title="비용 항목")

# ---------------- 3. 레이아웃 구성 ----------------
col_lcoh, col_fin = st.columns([1.4, 1])

# 3-1. 왼쪽: LCOH 분석
with col_lcoh:
    st.subheader("LCOH 분석")

    annual_elec_mwh = annual_elec_kwh / 1000.0
    annual_h2_ton = annual_h2_kg / 1000.0

    table_md = f"""
    <table style='font-size:18px; font-weight:500;'>
    <tr>
        <td style='padding: 6px 15px;'>LCOH (원/kgH₂)</td>
        <td style='padding: 6px 15px; font-weight:bold;'>{lcoh_krw_per_kg:,.0f}</td>
    </tr>
    <tr>
        <td style='padding: 6px 15px;'>연간 전력 사용량 (MWh/년)</td>
        <td style='padding: 6px 15px; font-weight:bold;'>{annual_elec_mwh:,.0f}</td>
    </tr>
    <tr>
        <td style='padding: 6px 15px;'>연간 수소 생산량 (Ton H₂/년)</td>
        <td style='padding: 6px 15px; font-weight:bold;'>{annual_h2_ton:,.0f}</td>
    </tr>
    </table>
    """
    st.markdown(table_md, unsafe_allow_html=True)

    # 상단: LCOH 비용 구성 파이차트 + 민감도 분석
    col_pie, col_sensi = st.columns(2)

    with col_pie:
        st.markdown("### LCOH 구성")
        total_for_share = sum(values)
        if total_for_share > 0:
            fig_pie = px.pie(names=drivers, values=values, color=drivers, color_discrete_map=colors)
            fig_pie.update_layout(font=dict(family="Arial Black", size=18), legend=dict(font=dict(size=16)))
            fig_pie.update_traces(textfont=dict(size=16))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("LCOH 계산을 위해 유효한 입력값을 먼저 넣어주세요.")

    with col_sensi:
        st.markdown("### 민감도 분석 (입력값 ±40%)")
        sensi_vars = ["에너지 효율", "전기요금", "CAPEX", "연간 운전시간", "OPEX", "할인율"]
        base_params = {
            "specific_energy": specific_energy,
            "elec_price": elec_price,
            "capex_total": capex_total,
            "annual_operating_hours": annual_operating_hours,
            "opex_annual": opex_annual,
            "discount_rate": discount_rate,
        }
        delta_minus, delta_plus = [], []

        for name in sensi_vars:
            # --- -40% 케이스 ---
            p = base_params.copy()
            if name == "에너지 효율":
                p["specific_energy"] *= 0.6
            elif name == "전기요금":
                p["elec_price"] *= 0.6
            elif name == "CAPEX":
                p["capex_total"] *= 0.6
            elif name == "연간 운전시간":
                p["annual_operating_hours"] *= 0.6
            elif name == "OPEX":
                p["opex_annual"] *= 0.6
            elif name == "할인율":
                p["discount_rate"] *= 0.6

            lcoh_m = compute_lcoh_given_params(p["capex_total"], p["specific_energy"], p["elec_price"], p["annual_operating_hours"], p["opex_annual"], p["discount_rate"])
            delta_minus.append(lcoh_m - lcoh_krw_per_kg)

            # --- +40% 케이스 ---
            p = base_params.copy()
            if name == "에너지 효율":
                p["specific_energy"] *= 1.4
            elif name == "전기요금":
                p["elec_price"] *= 1.4
            elif name == "CAPEX":
                p["capex_total"] *= 1.4
            elif name == "연간 운전시간":
                p["annual_operating_hours"] *= 1.4
            elif name == "OPEX":
                p["opex_annual"] *= 1.4
            elif name == "할인율":
                p["discount_rate"] *= 1.4

            lcoh_p = compute_lcoh_given_params(p["capex_total"], p["specific_energy"], p["elec_price"], p["annual_operating_hours"], p["opex_annual"], p["discount_rate"])
            delta_plus.append(lcoh_p - lcoh_krw_per_kg)

        max_change = max(max(abs(np.array(delta_minus))), max(abs(np.array(delta_plus))))

        fig_sensi = go.Figure()
        fig_sensi.add_bar(y=sensi_vars, x=delta_minus, orientation="h", name="-40%")
        fig_sensi.add_bar(y=sensi_vars, x=delta_plus, orientation="h", name="+40%")
        fig_sensi.update_layout(xaxis_title="ΔLCOH (원/kgH₂)", barmode="relative",
                                 font=dict(family="Arial Black", size=18), legend=dict(font=dict(size=16)))
        fig_sensi.update_xaxes(range=[-max_change * 1.1, max_change * 1.1], zeroline=True, tickfont=dict(size=16))
        fig_sensi.update_yaxes(tickfont=dict(size=16))
        st.plotly_chart(fig_sensi, use_container_width=True)
    
    st.markdown("### LCOH 구성요소")
    # 주요 비용 구성요소별 LCOH 기여 (원/kgH₂)
    capex_construction_per_kg = capex_construction_per_kg
    capex_equipment_per_kg = capex_equipment_per_kg
    opex_per_kg = opex_per_kg
    elec_per_kg = elec_per_kg

    # 누적 영역 계산
    c1 = capex_construction_per_kg
    c2 = c1 + capex_equipment_per_kg
    c3 = c2 + opex_per_kg
    c4 = c3 + elec_per_kg

    x_labels = ["Total LCOH", "CAPEX-공사", "CAPEX-설비", "OPEX-O&M", "OPEX-전력"]
    fig_major = go.Figure()
    fig_major.add_bar(name="CAPEX-공사", x=["Total LCOH"], y=[capex_construction_per_kg], base=[0], marker_color=colors["CAPEX-공사"])
    fig_major.add_bar(name="CAPEX-설비", x=["Total LCOH"], y=[capex_equipment_per_kg], base=[c1], marker_color=colors["CAPEX-설비"])
    fig_major.add_bar(name="OPEX-O&M", x=["Total LCOH"], y=[opex_per_kg], base=[c2], marker_color=colors["OPEX-O&M"])
    fig_major.add_bar(name="OPEX-전력", x=["Total LCOH"], y=[elec_per_kg], base=[c3], marker_color=colors["OPEX-전력"])
    # (개별 막대)
    fig_major.add_bar(name="CAPEX-공사 (개별)", x=["CAPEX-공사"], y=[capex_construction_per_kg], base=[0], marker_color=colors["CAPEX-공사"], showlegend=False)
    fig_major.add_bar(name="CAPEX-설비 (개별)", x=["CAPEX-설비"], y=[capex_equipment_per_kg], base=[c1], marker_color=colors["CAPEX-설비"], showlegend=False)
    fig_major.add_bar(name="OPEX (개별)", x=["OPEX-O&M"], y=[opex_per_kg], base=[c2], marker_color=colors["OPEX-O&M"], showlegend=False)
    fig_major.add_bar(name="전력비용 (개별)", x=["OPEX-전력"], y=[elec_per_kg], base=[c3], marker_color=colors["OPEX-전력"], showlegend=False)

    fig_major.update_layout(barmode="overlay", yaxis_title="원/kgH₂", xaxis_title="", 
                             font=dict(family="Arial Black", size=18), legend=dict(font=dict(size=18)))
    fig_major.update_xaxes(categoryorder="array", categoryarray=x_labels, tickfont=dict(size=18))
    fig_major.update_yaxes(tickfont=dict(size=18))
    st.plotly_chart(fig_major, use_container_width=True)

# 3-2. 오른쪽: 경제성 분석
with col_fin:
    st.subheader("경제성 분석")
    npv_million = npv / 1_000_000
    bc_str = f"{bc_ratio:,.2f}" if not np.isnan(bc_ratio) else "N/A"
    pirr_str = f"{p_irr_pct:,.2f}" if not np.isnan(p_irr_pct) else "N/A"
    fin_table = f"""
    <table style='font-size:18px; font-weight:500;'>
        <tr><td style='padding: 6px 15px;'>NPV (백만원)</td><td style='padding: 6px 15px; font-weight:bold;'>{npv_million:,.0f}</td></tr>
        <tr><td style='padding: 6px 15px;'>B/C 비율 (-)</td><td style='padding: 6px 15px; font-weight:bold;'>{bc_str}</td></tr>
        <tr><td style='padding: 6px 15px;'>P-IRR (%)</td><td style='padding: 6px 15px; font-weight:bold;'>{pirr_str}</td></tr>
    </table>
    """
    st.markdown(fin_table, unsafe_allow_html=True)
    st.markdown("### 유입/유출 현재가치 및 NPV")
    fin_labels = ["유입 현재가치", "유출 현재가치", "순현재가치(NPV)"]
    # npv_display = abs(npv)
    fin_values = [pv_in, pv_out, npv]
    fig_fin = go.Figure(data=[go.Bar(x=fin_labels, y=fin_values, text=[f"{v:,.0f}" for v in fin_values], textposition="auto")])
    fig_fin.update_layout(yaxis_title="원", font=dict(family="Arial Black", size=18), legend=dict(font=dict(size=16)))
    fig_fin.update_yaxes(range=[0, max(pv_in, pv_out) * 1.2])
    fig_fin.update_xaxes(tickfont=dict(size=16))
    fig_fin.update_yaxes(tickfont=dict(size=16))
    st.plotly_chart(fig_fin, use_container_width=True)

    # 투자회수기간 계산 및 그래프
    years = list(range(len(cash_flows)))
    cum_cf = np.cumsum(cash_flows)
    payback_year = None
    for i in range(1, len(years)):
        if cum_cf[i] >= 0:
            prev_cf = cum_cf[i-1]
            this_cf = cum_cf[i]
            if this_cf == prev_cf:
                payback_year = years[i]
            else:
                frac = -prev_cf / (this_cf - prev_cf)
                payback_year = years[i-1] + frac
            break

    st.markdown("### 투자회수기간")
    fig_pay = go.Figure()
    fig_pay.add_trace(go.Scatter(x=years, y=cum_cf, mode="lines+markers", name="누적 현금흐름"))
    fig_pay.add_shape(type="line", x0=years[0], x1=years[-1], y0=0, y1=0, line=dict(color="gray", dash="dash"))
    if payback_year is not None:
        fig_pay.add_shape(type="line", x0=payback_year, x1=payback_year, y0=min(cum_cf), y1=max(cum_cf), line=dict(color="red", dash="dot"))
        fig_pay.add_trace(go.Scatter(x=[payback_year], y=[0], mode="markers", marker=dict(size=10, color="red"), name="Payback point"))
    fig_pay.update_layout(xaxis_title="운영 연도", yaxis_title="누적 현금흐름 (원)", font=dict(family="Arial Black", size=18), legend=dict(font=dict(size=16)))
    fig_pay.update_xaxes(tickfont=dict(size=16))
    fig_pay.update_yaxes(tickfont=dict(size=16))
    st.plotly_chart(fig_pay, use_container_width=True)

